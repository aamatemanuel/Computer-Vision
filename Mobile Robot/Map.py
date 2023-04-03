import math
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_function_of_line(line):
    """
    returns parameters a and b so the line can be expressed as y = a*x+b
    """
    if (line.end.x - line.beginning.x) == 0:
        a = 10**10
    else:
        a = (line.end.y - line.beginning.y)/(line.end.x - line.beginning.x)
    b = line.beginning.y - a * line.beginning.x
    return a, b

def distance(a,b):
    """
    returns the dictance between two points
    """
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def is_point_on_line(point,line,threshold):
    """
    checks if a point is situated on a certain line
    """
    point_on_line = False
    if distance(line.beginning, point) + distance(line.end, point) - distance(line.beginning, line.end) <= threshold:
        point_on_line = True
    return point_on_line

def are_points_close(point1,point2,threshold):
    if distance(point1,point2) <= threshold:
        return True
    else:
        return False
class Point:
    """
    point class
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.reprz = str((x,y))

class Line:
    """
    Line class:
        - a line is defined by the two points that are connected by it
        - these points are situated within a cluster of points
        - the line angle is between -180 and 180 degrees
        - the line's neighbours are the other lines that share a cluster
    """
    def __init__(self, line):
        x1, y1, x2, y2 = line[0]
        delta_y = y2 - y1
        delta_x = x2 - x1
        self.angle = np.arctan2(delta_y, delta_x)
        while self.angle < 0:
            self.angle += math.pi
        while self.angle > math.pi:
            self.angle -= math.pi
        self.beginning = Point(x1, y1)
        self.end = Point(x2, y2)
        self.points = {self.beginning, self.end}

        self.beginningNeighbours = list()
        self.endNeighbours = list()
        self.beginningCluster = None
        self.endCluster = None

        self.reprz = str(line[0])

    def eq(self, other): # hello tuur #__eq__(self, other):
        if isinstance(other, Line):
            return self.beginningCluster == other.beginningCluster and self.endCluster == other.endCluster
        return False

    def hash(self): #__hash__(self):
        return hash((self.beginningCluster, self.endCluster))
    def hash2(self): #__hash__(self):
        return hash((self.endCluster, self.beginningCluster))

    def addToCluster(self, point, cluster):
        """
        line is added to a cluster
        """
        if point == self.beginning:
            self.beginningCluster = cluster
        elif point == self.end:
            self.endCluster = cluster
        else:
            print('wtfrick')
            quit()
    def angleDiff(self, line2):
        """
        return the difference in angle between two lines. Always expressed positive so a clockwise loop algorithm can be implemented
        """
        angle = self.angle - line2.angle
        if angle < 0:
            angle += 2*math.pi
        return angle

    def sortNeighbours(self):
        """
        sort neighbours in clockwise fashion, using helper function angleDiff
        """
        self.beginningNeighbours.sort(key=lambda x: self.angleDiff(x))
        self.endNeighbours.sort(key=lambda x: self.angleDiff(x))

class CornerCluster:
    """
    cluster of points that are situated near to each other.
    - a cluster always contains at least one point
    - a cluster's position is the average of the points within it
    """
    def __init__(self, point):
        self.lines = set()
        self.points = set()

        self.x = point.x
        self.y = point.y



    def reprz(self):
        """
        representation of the cluster
        """
        return str((self.x, self.y))

    def addLine(self, line, point):
        """
        - add a line to the cluster
        - recalculate cluster position
        """
        assert point in line.points
        #self.linesDict[point] = line
        self.lines.add(line)
        self.points.add(point)
        sum_x = 0
        sum_y = 0
        n = 0
        for point in self.points: #self.linesDict.keys():
            sum_x += point.x
            sum_y += point.y
            n += 1
        self.x = sum_x/n
        self.y = sum_y/n

    def removeLineFromCluster(self,line_remove):
        self.lines.remove(line_remove)



    def dist(self, point):
        return math.sqrt((self.x-point.x)**2+(self.y-point.y)**2)


class Loop:
    """
    a loop is a series of lines that are connected
    """
    def __init__(self, lines):#, clusters):
        self.lines = lines
        #self.clusters = clusters
    def reprz(self):
        i = 0
        print('loop of length: '+str(len(self.lines)))
        line = self.lines[i]
        print('goal: '+line.reprz + ' '+str(self))
        i+=1
        while i <= len(self.lines)-2:
            line = self.lines[i]
            print('next: '+line.reprz + ' '+str(self))
            i += 1
        line = self.lines[i]
        print('beginning: '+line.reprz + ' '+str(self))

class Graph:
    """
    in the end, the 2 dicts are not used I think, but they might still be useful later on
    """
    def __init__(self):
        self.lines = set()
        self.clusters = set()

        self.pointToLine = dict()
        self.clusterToLines = dict()

        self.loops = set()


    def addLineToGraph(self, rawline):
        """
        if the line contains a point that is situated within one of the clusters: it is added to that cluster
        if the point is not inside a cluster, a new cluster is created
        a point can only belong to one cluster
        all the graph's data structures are updated
        """
        print('new line:')

        if not isinstance(rawline, Line):
            print(rawline)
            line = Line(rawline)

        else:
            print(rawline.reprz)
            line = rawline
        self.lines.add(line)
        print(line.beginning.reprz)
        print(line.end.reprz)
        print(' ')
        TRESHOLD = 1
        for point in line.points:
            print('analysing point')
            print(point.reprz)

            self.pointToLine[point] = line
            foundCluster = False
            for cluster in self.clusters:
                if cluster.dist(point) < TRESHOLD:
                    cluster.addLine(line, point)
                    self.clusterToLines[cluster].add(line)
                    if point == line.beginning:
                        print('found beginning')
                        print(str([cluster.x, cluster.y]))
                        line.beginningCluster = cluster
                    else:
                        print('found end')
                        print(str([cluster.x, cluster.y]))
                        line.endCluster = cluster
                    foundCluster = True
                    break
            if not foundCluster:
                newCluster = CornerCluster(point)
                self.clusters.add(newCluster)
                newCluster.addLine(line, point)
                self.clusterToLines[newCluster] = set()
                self.clusterToLines[newCluster].add(line)
                if point == line.beginning:
                    print('created beginning')
                    print(str([newCluster.x, newCluster.y]))
                    line.beginningCluster = newCluster
                else:
                    print('created ending')
                    print(str([newCluster.x, newCluster.y]))
                    line.endCluster = newCluster
            print(' ')
        assert line.endCluster != None
        assert line.beginningCluster != None

    def addNeighbour(self, firstLine: Line, secondLine: Line, cluster: CornerCluster):
        """
        add the two lines as each other's neighbours, linking them to the right cluster
        """
        print(firstLine)
        print(secondLine)
        print(cluster)
        if cluster == firstLine.beginningCluster:
            firstLine.beginningNeighbours.append(secondLine)
        else:
            firstLine.endNeighbours.append(secondLine)
        if cluster == secondLine.beginningCluster:
            secondLine.beginningNeighbours.append(firstLine)
        else:
            secondLine.endNeighbours.append(firstLine)


    def findNeighbours(self):
        """
        go through all the lines to add all neighbours
        """
        for cluster in self.clusters:
            linesLeft = list(cluster.lines)
            for line in cluster.lines:
                linesLeft.remove(line)
                for line2 in linesLeft:
                    self.addNeighbour(line, line2, cluster)
        for line in self.lines:
            line.sortNeighbours()


    def findLoopsBackTrack(self):
        """
        initialise the backtracking algorithm that finds all loops
        """
        linesToExplore = list(self.lines)
        for line in linesToExplore:
            for firstPath in line.beginningNeighbours:
                # arg: current line, goal cluster, start cluster
                loop = self.tryLine(firstPath, line.endCluster, line.beginningCluster)
                if loop != None and line not in loop: # and firstPath not in loop:
                    loop.append(line)
                    self.loops.add(Loop(loop))

            # i think the code below only leads to duplicates
            #for firstPath in line.endNeighbours:
            #    loop = self.tryLine(firstPath, line.beginningCluster, line.endCluster)
            #    if loop != None:
            #        loop.append(line)
            #        self.loops.add(Loop(loop))

    def tryLine(self, current: Line, goal: CornerCluster, comesFrom: CornerCluster, depth=0):
        """
        recursively try to follow paths until the goal is reached or max depth is reached or we're stuck in a different loop
        """
        DEPTHMAX = 4
        #print('trying line '+current.reprz+' to get to '+goal.reprz())

        if depth >= DEPTHMAX:
            return None

        if comesFrom == current.endCluster:
            if current.beginningCluster == goal:
                return [current]
        elif comesFrom == current.beginningCluster:
            if current.endCluster == goal:
                return [current]

        if comesFrom == current.endCluster:
            #if current.beginningCluster == goal:
            #    return [current]
            for i in range(len(current.beginningNeighbours)):
                next = current.beginningNeighbours[i]
                # arg: current line, goal cluster, start cluster
                result = self.tryLine(next, goal, current.beginningCluster, depth+1)
                if result != None and current not in result:
                    result.append(current)
                    return result
            return None

        elif comesFrom == current.beginningCluster:
            #if current.endCluster == goal:
            #    return [current]
            for i in range(len(current.endNeighbours)):
                next = current.endNeighbours[i]
                result = self.tryLine(next, goal, current.endCluster, depth+1)
                if result != None and current not in result:
                    result.append(current)
                    return result
            return None

    def removeDuplicateLoops(self, loopsToCheck=None):
        """
        go through all the loops and remove duplicates
        """
        print(' ')
        print('====')

        if loopsToCheck is None:
            loopsToCheck = self.loops.copy()
        loopsUnchecked = loopsToCheck.copy()

        print(loopsToCheck)
        print(self.loops)

        for loop in loopsToCheck:
            loopsUnchecked.remove(loop)
            isADuplicate = self.checkDup(loop, loopsUnchecked)

            if isADuplicate:
                print('-- removing loop --')

                self.loops.remove(loop)
                #loopsUnchecked.remove(loop)

            if not isADuplicate: # restart because
                # wow i dont need this anymore :))
                # So the
                """ to be honest i changed so much stuff to make it work that i dont even know how/why it works :)) """
                #loopsUnchecked.remove(loop)
                #break
                pass

        # can I delete this?
        if len(loopsUnchecked) > 1 and loopsUnchecked != loopsToCheck:
            print('-- used recursion in removeDuplicates --')
            self.removeDuplicateLoops(loopsUnchecked)

    def checkDup(self, loop: Loop, loopsToCheck):
        """
        helper function for removeduplicates to check if the loop has a duplicate in loopsToCheck
        the loops are duplicates if they contain the same lines
        every line in a loop is unique, so i only have to check in one direction
        """
        nbLines = len(loop.lines)
        for secondLoop in loopsToCheck:
            #if loop != secondLoop: # loopsToCheck still contains "loop" itself !!
            if True:
                if nbLines == len(secondLoop.lines):
                    matches = 0
                    for line in loop.lines:
                        if line in secondLoop.lines:
                            matches += 1
                    if matches == nbLines:
                        return True
        return False


    def displayGraph(self):
        """
        display the results
        """

        plt.figure()
        plt.title('original graph')
        for line in self.lines:
            plt.plot([line.beginning.x, line.end.x], [line.beginning.y, line.end.y])
        plt.show()
        #time.sleep(2)
        plt.figure()
        plt.title('graph after clustering')
        for line in self.lines:
            plt.plot([line.beginningCluster.x, line.endCluster.x], [line.beginningCluster.y, line.endCluster.y])
        plt.show()
        #time.sleep(2)

        loopid = 0
        for loop in self.loops:
            plt.figure()
            plt.title('detected loop: '+str(loopid) + ' with length: '+str(len(loop.lines)))
            for line in loop.lines:
                plt.text(0.5, 0.5, "str(line)")
                plt.plot([line.beginningCluster.x, line.endCluster.x], [line.beginningCluster.y, line.endCluster.y])
            plt.show()
            loopid += 1

    plt.figure()
    plt.title('detected loops')

    def removeDuplicateLines(self):
        '''
        sets automatically delete duplicates, so a new set is created with the elements from the older set.
        Objects are compared by there hash-value, which is in this case (beginning_cluster,end_cluster).
        The same is done for the set of lines in each cluster from the graph
        '''
        linesWithoutDups = set()
        hashedLines = set()
        removedLines = set()
        for line in self.lines:
            hsh = line.hash()
            hsh2 = line.hash2()
            if hsh not in hashedLines and hsh2 not in hashedLines and line.beginningCluster != line.endCluster:
                hashedLines.add(hsh)
                hashedLines.add(hsh2)
                linesWithoutDups.add(line)
            else:
                removedLines.add(line)
        self.lines = linesWithoutDups

        for cluster in self.clusters:
            cluster_lines_copy = set()
            for line in cluster.lines:
                if line not in removedLines:
                    cluster_lines_copy.add(line)
            cluster.lines = cluster_lines_copy

    def removeLineFromGraph(self,line):

        #print('removing line:')
        #print(self.lines)
        #print(line)
        #for line2 in self.lines.copy():
        #    if line == line2:
        #        self.lines.remove(line2)
        #        line2.beginningCluster.lines.remove(line2)
        #        line2.endCluster.lines.remove(line2)
        #        print(line2)
        #        print(True)
        #    else:
        #        print(False)

        assert line in self.lines
        self.lines.remove(line)
        print(line.reprz)
        print(line.beginning)
        print(line.beginningCluster)
        print(line.end)
        print(line.endCluster)

        line.beginningCluster.lines.remove(line)
        if line in line.endCluster.lines:
            line.endCluster.lines.remove(line)

    def displayGraphResult(self, warped_img):
        for loop in self.loops:
            for line in loop.lines:
                cv2.line(warped_img, ( int(line.beginningCluster.x), int(line.beginningCluster.y) ), (int(line.endCluster.x), int(line.endCluster.y)), (255, 255, 0), 1)
        cv2.imshow('the loops found by in the graph', warped_img)

    def splitIntersectingLines(self, depth=0):
        cross_sec_x = None
        cross_sec_y = None
        new_line1 = Line([[0,0,5,2]])
        new_line2 = Line([[0,0,5,2]])
        new_line3 = Line([[0,0,5,2]])
        new_line4 = Line([[0,0,5,2]])

        threshold_for_closeness = 0.0001
        lines_in_graph = list(self.lines)
        removedSomething = False
        i = 0
        while i  < len(lines_in_graph)-1 and not removedSomething:
            line = lines_in_graph[i]
            i+=1
            j = i
            while j < len(lines_in_graph)-1 and not removedSomething:
                line_comp = lines_in_graph[j]
                j+=1
                if line_comp == line:
                    continue
                line_slope,line_bias = get_function_of_line(line)
                line_comp_slope, line_comp_bias = get_function_of_line(line_comp)

                """ if not parallel """
                TRESHOLDLINESLOPE = 0.001
                if (line_slope - line_comp_slope) > TRESHOLDLINESLOPE:
                    print('')
                    print('cross section: ')
                    print(line_slope)
                    cross_sec_x = (line_bias - line_comp_bias) / (line_comp_slope - line_slope)
                    cross_sec_y = line_slope*cross_sec_x + line_bias
                    cross_sec = Point(cross_sec_x, cross_sec_y)
                    print(cross_sec.reprz)

                    print("")
                    """ if not crossing """
                    print((is_point_on_line(cross_sec,line,threshold_for_closeness) and is_point_on_line(cross_sec,line_comp,threshold_for_closeness)))
                    if not (is_point_on_line(cross_sec,line,threshold_for_closeness) and is_point_on_line(cross_sec,line_comp,threshold_for_closeness)):
                        continue
                    # if points intersection and beginning or end are close
                    elif are_points_close(cross_sec,line.beginning,threshold_for_closeness) or are_points_close(cross_sec,line.end,threshold_for_closeness):

                        if are_points_close(cross_sec,line_comp.beginning,threshold_for_closeness) or are_points_close(cross_sec,line_comp.end,threshold_for_closeness):
                            continue
                        else:

                            new_line3.beginning = line_comp.beginning
                            new_line3.end = cross_sec
                            new_line3.points = {line_comp.beginning, cross_sec}
                            new_line3.reprz = str(
                                [new_line3.beginning.x, new_line3.beginning.y, new_line3.end.x, new_line3.end.y])
                            new_line4.beginning = cross_sec
                            new_line4.end = line_comp.end
                            new_line4.points = {line_comp.end, cross_sec}
                            new_line4.reprz = str(
                                [new_line4.beginning.x, new_line4.beginning.y, new_line4.end.x, new_line4.end.y])

                            self.addLineToGraph(new_line3)
                            self.addLineToGraph(new_line4)
                            self.removeLineFromGraph(line_comp)
                            removedSomething = True
                            print('case1')
                    # if points intersection and beginning or end are close
                    elif are_points_close(cross_sec,line_comp.beginning,threshold_for_closeness) or are_points_close(cross_sec,line_comp.end,threshold_for_closeness):
                        if are_points_close(cross_sec,line.beginning,threshold_for_closeness) or are_points_close(cross_sec,line.end,threshold_for_closeness):
                            continue
                        else:
                            new_line3.beginning = line.beginning
                            new_line3.end = cross_sec
                            new_line3.points = {line.beginning, cross_sec}
                            new_line3.reprz = str(
                                [new_line3.beginning.x, new_line3.beginning.y, new_line3.end.x, new_line3.end.y])
                            new_line4.beginning = cross_sec
                            new_line4.end = line.end
                            new_line4.points = {line.end, cross_sec}
                            new_line4.reprz = str(
                                [new_line4.beginning.x, new_line4.beginning.y, new_line4.end.x, new_line4.end.y])

                            self.addLineToGraph(new_line3)
                            self.addLineToGraph(new_line4)
                            self.removeLineFromGraph(line)
                            removedSomething = True
                            print('case2')

                    elif are_points_close(cross_sec,line.end,threshold_for_closeness) and (are_points_close(cross_sec,line_comp.beginning,threshold_for_closeness) or are_points_close(cross_sec,line_comp.end,threshold_for_closeness)):
                            continue
                    elif are_points_close(cross_sec,line.beginning,threshold_for_closeness) and (are_points_close(cross_sec,line_comp.beginning,threshold_for_closeness) or are_points_close(cross_sec,line_comp.end,threshold_for_closeness)):
                            continue
                    else:
                        print('entering case 3')
                        print(line.reprz)
                        print(line_comp.reprz)
                        print(" ")
                        new_line1.beginning = line.beginning
                        new_line1.end = cross_sec
                        new_line1.points = {line.beginning, cross_sec}
                        new_line1.reprz = str(
                            [new_line1.beginning.x, new_line1.beginning.y, new_line1.end.x, new_line1.end.y])

                        new_line2.beginning = cross_sec
                        new_line2.end = line.end
                        new_line2.points = {line.end, cross_sec}
                        new_line2.reprz = str(
                            [new_line2.beginning.x, new_line2.beginning.y, new_line2.end.x, new_line2.end.y])

                        new_line3.beginning = line_comp.beginning
                        new_line3.end = cross_sec
                        new_line3.points = {line_comp.beginning, cross_sec}
                        new_line3.reprz = str(
                            [new_line3.beginning.x, new_line3.beginning.y, new_line3.end.x, new_line3.end.y])

                        new_line4.beginning = cross_sec
                        new_line4.end = line_comp.end
                        new_line4.points = {line_comp.end, cross_sec}
                        new_line4.reprz = str(
                            [new_line4.beginning.x, new_line4.beginning.y, new_line4.end.x, new_line4.end.y])

                        self.addLineToGraph(new_line1)
                        self.addLineToGraph(new_line2)
                        self.addLineToGraph(new_line3)
                        self.addLineToGraph(new_line4)

                        self.removeLineFromGraph(line)
                        self.removeLineFromGraph(line_comp)
                        removedSomething = True
                        print('case3')

                else:
                    continue
        if removedSomething:
            #self.displayGraph()
            print(' ')
            print('restart')
            print("depth:" + str(depth))
            self.splitIntersectingLines(depth+1)
        else:
            self.displayGraph()

    def addEdgdesAsLines(self, xwidthtop, xwidthbottom, xwidthtotal,  ymax, ymin):
        xcentre = round(xwidthtotal/2)
        xbottomLeft = round(xcentre - xwidthbottom/2)
        xbottomRight = round(xcentre + xwidthbottom / 2)
        xtopleft = round(xcentre - xwidthtop/2)
        xtopright = round(xcentre + xwidthtop/2)

        ybottomleft = ymin
        ybottomright = ymin
        ytopleft = ymax
        ytopright = ymax

        edge1 = Line([[xbottomLeft,ybottomleft,xbottomRight,ybottomright]])
        edge2 = Line([[xbottomLeft,ybottomleft,xtopleft,ytopleft]])
        edge3 = Line([[xbottomRight,ybottomright,xtopright,ytopright]])
        edge4 = Line([[xtopleft,ytopleft,xtopright,ytopright]])

        self.addLineToGraph(edge1)
        self.addLineToGraph(edge2)
        self.addLineToGraph(edge3)
        self.addLineToGraph(edge4)

def getAreaOfLoop(loop): #works but not for triangles-> have to adapt it
    corners = set()
    for line in loop.lines:
        corners.add(line.beginningCluster)
        corners.add(line.endCluster)
    corners = list(corners)
    x1 = corners[0].x
    y1 = corners[0].y
    x2 = corners[1].x
    y2 = corners[1].y
    x3 = corners[2].x
    y3 = corners[2].y
    x4 = corners[3].x
    y4 = corners[3].y
    A = 0.5*np.abs((x3*y2 - x2*y3) + (x4*y3 - x3*y4) + (x2*y1 - x1*y2) + (x1*y4 - x4*y1))
    return A



test = 0
if test:

    a = [[1,2,3.1,4.1]]
    b = [[2.8,4,2,0]]
    c = [[3, 4, 4, 10]]
    d = [[4, 10, 1, 2]]
    e = [[10.2, 9.9, 3,4]]
    f = [[11, 12, 1, 4.2]]
    g = [[0.3, 0.1, 4, 3]]
    h = [[0, 0, 10, 10]]
    i = [[12, 4, 7, 9]]
    j = [[10,10.5, 7, 9]]
    k = [[7, 9, 3, 4]]
    #l = [[7.001, 9.001, 3.001, 4.001]]
    m = [[5,5,5,10]]
    n = [[5.1,5.1, 10, 5]]
    o = [[0,7.5,7.5,7.5]]


    print('-- lines --')
    print('a: '+ str(a))
    print('b: ' + str(b))
    print('')


    start = time.time()

    myMap = Graph()
    myMap.addLineToGraph(a)
    myMap.addLineToGraph(b)
    myMap.addLineToGraph(c)
    myMap.addLineToGraph(d)
    myMap.addLineToGraph(e)
    myMap.addLineToGraph(f)
    myMap.addLineToGraph(g)
    myMap.addLineToGraph(h)
    myMap.addLineToGraph(i)
    myMap.addLineToGraph(j)
    myMap.addLineToGraph(k)
    #myMap.addLineToGraph(l)
    myMap.addLineToGraph(m)
    myMap.addLineToGraph(n)
    myMap.addLineToGraph(o)

    # show initial
    myMap.displayGraph()

    # remove all duplicates
    myMap.removeDuplicateLines()

    # show initial without duplicate lines
    myMap.displayGraph()

    # split all lines that intersect, this may create new duplicates
    myMap.splitIntersectingLines()
    myMap.removeDuplicateLines()

    # lines that share a cluster are neighbours
    myMap.findNeighbours()
    initialisationComplete = time.time()
    time_spent_on_initialisation = initialisationComplete-start

    # find loops through backtracking & remove duplicate loops
    myMap.findLoopsBackTrack()
    myMap.removeDuplicateLoops()

    loopsComplete = time.time()
    time_spent_on_loops = loopsComplete-initialisationComplete

    # display final result
    myMap.displayGraph()

    # for loop in myMap.loops:
    #     print(getAreaOfLoop(loop))
    # print('-- clusters --')
    # print(myMap.clusters)
    # print('')

    for cluster in myMap.clusters:
        print('-- cluster --')
        print('x: '+ str(cluster.x))
        print('y: '+ str(cluster.y))
        print('')


    for line in myMap.lines:
        print(' ')
        print('-- line --')
        print('line: ' + line.reprz)
        print('angle: '+str(line.angle))

        print('beginning: ' + line.beginning.reprz)
        print('beginning cluster: ' + line.beginningCluster.reprz())
        print('beginning neighbours: ' + str(line.beginningNeighbours))
        for neighbour in line.beginningNeighbours:
            print(neighbour.reprz)

        print('end: ' + line.end.reprz)
        print('end cluster: ' + line.endCluster.reprz())
        print('end neighbours: ' + str(line.endNeighbours))
        for neighbour in line.endNeighbours:
            print(neighbour.reprz)

    for loop in myMap.loops:
        print(' ')
        #print('loop: '+str(loop.lines))
        loop.reprz()


    print(' ')
    print('graph took '+str(time_spent_on_initialisation)+'s')
    print('loops took ' + str(time_spent_on_loops) + 's')


    # HELLO TUUR

