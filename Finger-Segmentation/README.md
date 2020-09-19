
DETECTING FINGERS :

1.Defining an ROI where fingers will be detected.

2.Running background analysis(Getting average background value) to study the background so that elements in the backgrounds would not be mistook as fingers.

3.Finding the largest contour in the ROI assuming it to be the hand.

4.Obtaining the topmost , rightend and leftend (extreme points) and finding the co-ordinates of the center of the hand by using convexHull() method.

5.Finding the euclidean distances between the center and the extreme points and storing the maximum one.

5.Creating a circular ROI with radius being 90% of the maximum euclidean distance and masking it on the thresholded version.

6.All the external contours are then detected in the imaginary circular ROI and contours which lie in the limit_points(circumference of the circular ROI) and out_of_wrist (Height greater than wrist points) are detected as fingers and no.of finger counters are counted hence giving the no.of fingers.
