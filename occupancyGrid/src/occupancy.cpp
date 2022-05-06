/**
**  Simple ROS Node
**/
#include <ros/ros.h>

void convertToOccupancy(img){
    int height = img
}
def convertToOccupancy(self, img):
        # function to take in the final lane detected image and convert into an occupancy grid
        height, width = img.shape[:2]
        resolution = 0.05 
        global grid
        grid = OccupancyGrid()
        m = MapMetaData()
        m.resolution = resolution
        m.width = width
        m.height = height
        pos = np.array([-width * resolution / 2, -height * resolution / 2, 0])
        m.origin = Pose()
        m.origin.position.x, m.origin.position.y = pos[:2]
        grid.info = m
        grid.data = self.convertImgToOgrid(img)
        

    def convertImgToOgrid(self, img):
        #function to take a cv2 image and return an int8[] array
        info = []
        for col in img:
            for row in col:
                val = np.uint8(row)
                if val[0] == 0:
                    info.append(0)
                elif val[0] == 100:
                    info.append(-1)
                elif val[0] == 255:
                    info.append(100)
                # rospy.loginfo(val[0])
        
        return info

int main(int argc, char* argv[])
{

  // This must be called before anything else ROS-related
  ros::init(argc, argv, "vision_node");

  // Create a ROS node handle
  ros::NodeHandle nh;

  ROS_INFO("Hello, World!");

  // Don't exit the program.
  ros::spin();
}