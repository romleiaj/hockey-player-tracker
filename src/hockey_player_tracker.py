#!/usr/bin/env python

'''
Lucas-Kanade homography hockey-player tracker
===============================

Lucas-Kanade sparse optical flow script. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames. Finds homography between reference and current views
and uses RANSAC to determine outliers. Clusters outliers using mean-
shift algorithm

Usage
-----
hockey_player_tracker.py [<video_source>]


Keys
----
Space   -   Pause
'''

# For Python 2/3 compatibility
from __future__ import print_function

import os
import time
import numpy as np
import cv2 as cv
import video
from common import draw_str
from video import presets
from collections import defaultdict
from sklearn.cluster import MeanShift, estimate_bandwidth


lk_params = dict( winSize  = (19, 19),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 5000,
                       qualityLevel = 0.01,
                       minDistance = 1,
                       blockSize = 19)

def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status

green = (0, 255, 0)
red = (0, 0, 255)

class App:
    def __init__(self, video_src, frame_offset):
        ### Stream info ###
        self.cam = video.create_capture(video_src, presets['book'])
        self.out = cv.VideoWriter('hockey_example.mp4', 
                cv.VideoWriter_fourcc(*'mp4v'), 20.0, (1008, 416))
        self.video = os.path.basename(video_src).split('.')[0]
        self.fps = self.cam.get(cv.CAP_PROP_FPS)
        ### Shared images and feature points ###
        self.p0 = None
        self.vis = None
        self.prev_gray = None
        ### Custom counters and variables ###
        self.i = 0
        self.frame_counter = 0
        self.frame = []
        self.pause = False
        self.skip = False
        self.total = []
        self.frame_offset = frame_offset
        # Squared pixels per-second delta allowed (CUSTOM)
        self.VEL_THRESH = 1.0         
        # RANSAC inlier/outlier threshold
        self.OUTLIER_THRESH = 2.0        
        # should be between [0, 1] 0.5 means that the median of all pairwise distances is used.
        self.QUANTILE = 0.271  # 
        # The number of samples to use in meanshift
        self.N_SAMPLES = 10
        # Frame buffer for feature collection (CUSTOM)
        self.FRAME = 1
        # Brightness threshold for outliers (CUSTOM)
        self.BRIGHTNESS_THRESH = 150

    def run(self):
        outliers = []
        inliers = []
        while True:
            start = time.time()
            if not self.pause or self.skip:
                _ret, self.frame = self.cam.read()
            if _ret == False:
                print("Exiting")
                return
            frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
            self.create_color_mask(frame_gray)
            self.vis = self.frame.copy()
            if self.p0 is not None:# and self.i == self.frame_offset:
                p2, trace_status = checkedTrace(self.prev_gray, frame_gray, self.p1)
                self.p1 = p2[trace_status].copy()
                self.p0 = self.p0[trace_status].copy()
                self.prev_gray = frame_gray
                # Need at least 4 points for clustering
                if len(self.p0) < 4:
                    self.p0 = None
                    continue
                H, status = cv.findHomography(self.p0, self.p1,
                        cv.RANSAC, self.OUTLIER_THRESH)
                for (x0, y0), (x1, y1), good in zip(self.p0[:,0], self.p1[:,0], status[:,0]):
                    dy = (y1 - y0)
                    dx = (x1 - x0)
                    # To throw away huge magnitudes have < 200
                    if good and (dx**2 + dy**2) < 200:
                        inliers.append(np.array([dx, dy]))
                    if not good and (dx**2 + dy**2) < 200:
                            outliers.append(np.array([x1, y1, dx, dy]))
                    # Can draw all features here
                    #cv.circle(self.vis, (x1, y1), 2, (red, green)[good], -1)
                self.cluster_outliers(inliers, outliers)
                if True: #self.frame_counter % self.FRAME == 0:
                    outliers = []
                    inliers = []
                draw_str(self.vis, (20, 20), 'track count: %d' % len(self.p1))
                draw_str(self.vis, (20, 40), 'RANSAC')
                self.i = 0
            else:
                p = cv.goodFeaturesToTrack(frame_gray, **feature_params)
            try:
                cv.imshow('Hockey Tracker', self.vis)
                self.out.write(self.vis)
            except cv.error:
                pass
            ch = cv.waitKey(1)
            if ch == 27:
                break
            if ch == ord(' '):
                self.pause = not self.pause
            if self.i == 0:
                self.frame0 = self.frame.copy()
                if not self.pause or self.skip:
                    # Generate new features
                    self.p0 = cv.goodFeaturesToTrack(frame_gray, **feature_params)
                if self.p0 is not None and not self.pause or self.skip:
                    self.p1 = self.p0
                    self.prev_gray = frame_gray
            self.i += 1
            self.frame_counter += 1
            self.total.append(time.time() - start)

    def cluster_outliers(self, background, foreground):
        trimmed_outliers = []
        # Calculate background movement from camera motion
        avg_dx = np.mean([pt[0] for pt in background])
        avg_dy = np.mean([pt[1] for pt in background])
        for pt in foreground:
            h, w = self.mask.shape[:2]
            x, y = map(int, pt[:2])
            dx, dy = pt[2:]
            y = h - 1 if y >= h else y
            x = w - 1 if x >= w else x
            # Checking to ensure each red point is within velocity threshold
            # and that each point when placed in the mask is bright (>150)
            if ((avg_dy - dy)**2 + (avg_dx - dx)**2) > self.VEL_THRESH \
                    and sum(self.mask[y][x]) > self.BRIGHTNESS_THRESH:
                cv.circle(self.vis, (x, y), 2, red, -1)
                trimmed_outliers.append(pt)
        print("Number of outliers trimmed: %s" % (len(foreground) - len(trimmed_outliers)))
        if len(trimmed_outliers) > 5:
            self.skip = False
            trimmed_outliers = np.array(trimmed_outliers)
            cluster_dict = defaultdict(list)
            # Creating clusters based on the trimmed outliers
            number, centers, labels = self.mean_shift(trimmed_outliers[:, :2])
            for i, label in enumerate(labels):
                # Grouping clusters of points
                cluster_dict[label].append(trimmed_outliers[i])
            if len(centers) > 0:
                for l, center in enumerate(centers):
                    if len(cluster_dict[l]) < 5:
                        continue
                    # Drawing arrows origin at cluster centers
                    x, y = map(int, center)
                    all_x = [pt[2] for pt in cluster_dict[l]]
                    all_y = [pt[3] for pt in cluster_dict[l]]
                    dx = np.mean(all_x) - avg_dx
                    dy = np.mean(all_y) - avg_dy
                    vel = np.sqrt(dx**2 + dy**2)/(1/self.fps)
                    try:
                        x1 = int(x + 10*dx)
                        y1 = int(y + 10*dy)
                    except ValueError:
                        x1 = x
                        y1 = y
                    cv.arrowedLine(self.vis, (x, y), (x1, y1), (100, 255, 100), 5)
                    if self.pause:
                        self.draw_vel(self.vis, (x-50, y+20), "%.2fpx/s" % vel)
        else:
            self.skip = True 

    def draw_vel(self, dst, target, s):
        x, y = target
        cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 
                1.0, (0, 0, 0), thickness = 6, lineType=cv.LINE_AA)
        cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 
                1.0, (255, 255, 255), lineType=cv.LINE_AA)


    def create_color_mask(self, frame_gray):
        black = np.zeros((frame_gray.shape[0], frame_gray.shape[1], 3), np.uint8)
        ret, thresh = cv.threshold(frame_gray, 140, 255, cv.THRESH_BINARY)
        _, contours, heirarchy = cv.findContours(thresh, cv.RETR_EXTERNAL,
                                                 cv.CHAIN_APPROX_NONE)
        cv.drawContours(black, contours, -1, (255, 255, 255), thickness= -1)
        self.mask = cv.blur(black, (50, 50))

    # #############################################################################
    # Compute clustering with MeanShift
    def mean_shift(self, points):
        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(points, quantile=self.QUANTILE,
                                       n_samples=self.N_SAMPLES)
        bandwidth += 0.01  # to ensure it's not 0

        ms = MeanShift(bandwidth=bandwidth,
                bin_seeding=True, cluster_all=False)
        ms.fit(points)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
    
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
    
        print("Number of estimated clusters : %d" % n_clusters_)
    
        return n_clusters_, cluster_centers, labels



def main():
    import sys
    try:
        video_src = str(sys.argv[1])
    except:
        video_src = 0
    # video_src = "Womens_hockey.avi"
    # optional frame skip frame_offset = int(sys.argv[2])
    frame_offset = 0

    print(__doc__)
    # Optional data write out for external analysis
    #with open('mag2_and_theta.csv','w') as of:
        #writer= csv.writer(of)
        #writer.writerow(["Mag-Squared | Theta | x0 | y0 | x1 | y1 | outlier(0):inlier(1)"])
    app = App(video_src, frame_offset)
    app.run()
    print("Avg FPS = %s" % (1/np.mean(app.total)))
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
