#!/usr/bin/env python3
"""ROS 2 node: listens to the microphone and publishes std_msgs/Bool True on
`alarm_heard` when the alarm in the template is recognised.

    source <your ROS / turtlebot3_ws setup>.bash
    python3 alarm_node.py --ros-args -p template:=alarm_template.npz -p device_name:="USB Composite"

Parameters
    template                (string, required) .npz from build_template.py
    device_name             (string) pick the first input device whose name contains this
                            (more reliable than the index, which can change between boots)
    device                  (int, default -1) input device index; used if device_name is empty,
                            -1 = system default
    sim_thr / margin_db / window_s   (float, 0 = use the template's value)
    rearm_s                 (float, 5.0) alarm must be gone this long before it can trigger again
    latch                   (bool, false) transient-local QoS: late subscribers also get the last message
    publish_false_on_clear  (bool, false) also publish False once the alarm has stopped
    debug                   (bool, false) once a second: in-band level vs noise floor, best similarity
                            vs the threshold, % of frames matched, buzzes found in the window

The first 5 s after start-up are used to measure the noise floor; no detection happens then.
True is published once per alarm episode (the alarm loops, but you only get one message).
"""
import queue

import numpy as np
import rclpy
import sounddevice as sd
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

import final.alarm_dsp as alarm_dsp


class AlarmNode(Node):
    def __init__(self):
        super().__init__("alarm_detector")
        self.declare_parameter("template", "")
        self.declare_parameter("device_name", "")
        self.declare_parameter("device", -1)
        self.declare_parameter("sim_thr", 0.0)
        self.declare_parameter("margin_db", 0.0)
        self.declare_parameter("window_s", 0.0)
        self.declare_parameter("rearm_s", 5.0)
        self.declare_parameter("latch", False)
        self.declare_parameter("publish_false_on_clear", False)
        self.declare_parameter("debug", False)

        def p(name):
            return self.get_parameter(name).value

        path = p("template")
        if not path:
            raise RuntimeError("Set the template parameter: -p template:=alarm_template.npz")
        tpl = alarm_dsp.load_template(path)
        self.det = alarm_dsp.Detector(tpl, sim_thr=p("sim_thr"), margin_db=p("margin_db"),
                                      window_s=p("window_s"), rearm_s=p("rearm_s"))
        self.publish_false = bool(p("publish_false_on_clear"))
        self.debug = bool(p("debug"))

        qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL if p("latch") else DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(Bool, "alarm_heard", qos)

        self.q = queue.Queue()
        self.stream = None
        self.n_overflow = 0
        self.stream = self._open_stream(int(tpl["sr"]), p("device_name"), int(p("device")))
        self.create_timer(0.02, self.on_timer)
        self.next_debug = 0.0
        self.get_logger().info(
            "Listening at %d Hz (sim_thr %.2f, %d buzzes in %.1fs). Measuring noise floor for %.0fs..."
            % (int(tpl["sr"]), self.det.sim_thr, self.det.n_required, self.det.window_s,
               self.det.warmup_s))

    # ---- audio
    def _find_device(self, name, index):
        if name:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
                    return i
            raise RuntimeError("No input device with %r in its name; run: python3 -m sounddevice" % name)
        return None if index < 0 else index

    def _open_stream(self, sr, name, index):
        dev = self._find_device(name, index)

        def callback(indata, frames, time_info, status):
            if status.input_overflow:
                self.n_overflow += 1
            self.q.put(indata[:, 0].copy())

        try:
            s = sd.InputStream(samplerate=sr, channels=1, dtype="int16", device=dev,
                               blocksize=self.det.f.hop, callback=callback)
            s.start()
        except sd.PortAudioError as e:
            raise RuntimeError("Could not open the microphone at %d Hz (device %r): %s" % (sr, dev, e))
        self.get_logger().info("Microphone: %s" % sd.query_devices(dev, "input")["name"])
        return s

    # ---- processing
    def publish_bool(self, value):
        msg = Bool()
        msg.data = bool(value)
        self.pub.publish(msg)

    def on_timer(self):
        chunks = []
        try:
            while True:
                chunks.append(self.q.get_nowait())
        except queue.Empty:
            pass
        if not chunks:
            return
        for ev in self.det.feed(np.concatenate(chunks)):
            if ev == "heard":
                self.publish_bool(True)
                self.get_logger().info("Alarm heard -> published True on alarm_heard")
            elif ev == "cleared":
                self.get_logger().info("Alarm gone; detector re-armed")
                if self.publish_false:
                    self.publish_bool(False)
        if self.debug and self.det.time_s >= self.next_debug:
            self.next_debug = self.det.time_s + 1.0
            m = self.det.summary()
            self.get_logger().info(
                "level %.1f dB (noise %.1f)  best sim %.2f (needs %.2f)  matched %3d%%  "
                "buzzes %d/%d  overflows %d"
                % (m["level"], m["noise"], m["best_sim"], self.det.sim_thr, 100 * m["matched"],
                   m["buzzes"], self.det.n_required, self.n_overflow))

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


def main():
    rclpy.init()
    node = AlarmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()