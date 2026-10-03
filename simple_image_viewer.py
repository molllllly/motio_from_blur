#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simple_image_viewer.py

Subscribe to one or more Image topics and display them with cv2.imshow.
Alternative to rqt_image_view when rqt is not available (e.g. Mac).

Usage:
  python2.7 simple_image_viewer.py /smear_path_vis /sparse_depth_vis
  python2.7 simple_image_viewer.py /blur_color

Press 'q' in any window to quit.
"""

from __future__ import print_function
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Viewer(object):
    def __init__(self, topics):
        self.bridge = CvBridge()
        self.imgs = {}
        self.subs = []
        for t in topics:
            # Each topic gets its own subscriber that writes into self.imgs.
            sub = rospy.Subscriber(
                t, Image,
                lambda msg, topic=t: self._on_image(topic, msg),
                queue_size=1)
            self.subs.append(sub)
            rospy.loginfo("Subscribed to %s", t)

    def _on_image(self, topic, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.imgs[topic] = img
        except Exception as e:
            rospy.logwarn("decode fail on %s: %s", topic, str(e))

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            for topic, img in list(self.imgs.items()):
                cv2.imshow(topic, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            rate.sleep()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: %s <topic1> [topic2] ..." % sys.argv[0])
        sys.exit(1)
    topics = [a for a in sys.argv[1:] if not a.startswith("_")]
    rospy.init_node("simple_image_viewer", anonymous=True)
    v = Viewer(topics)
    v.spin()
