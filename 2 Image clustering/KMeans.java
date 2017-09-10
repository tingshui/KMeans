/*** Author :Vibhav Gogate
The University of Texas at Dallas
*****/
/*----------------------------------------------------------------------
 *  Author: Qianying Lin
 *  Written: 4/7/2016
 *  Last updated: 4/7/2016
 * 
 *  Complication: 
 *  Execution: 
 *
 *  This file perform the kmean clustering.
 *---------------------------------------------------------------------*/


import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;
 

public class KMeans {
    public static void main(String [] args){
	if (args.length < 3){
	    System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
	    return;
	}
	
	int iteration = 10;
	long ori_size = 0;
	long[] comp_size = new long[iteration];
	double[] comp_rate = new double[iteration];
	double comp_mean = 0;
	double comp_var = 0;
	int k = 0;
	for (int i = 0; i < iteration; i++) {
		try{
			File ori = new File(args[0]);
			BufferedImage originalImage = ImageIO.read(ori);
			ori_size = ori.length();
			k=Integer.parseInt(args[1]);
			BufferedImage kmeansJpg = kmeans_helper(originalImage,k);
			File out = new File(args[2]);
			ImageIO.write(kmeansJpg, "jpg", out);
			comp_size[i] = out.length();	    
		}catch(IOException e){
			System.out.println(e.getMessage());
		}	
	}
	double sum = 0;
	for (int i = 0; i < iteration; i++) {
		comp_rate[i] = ((double)ori_size/(double)comp_size[i]);
		sum += comp_rate[i];
	}
	comp_mean = sum/iteration;
	double sum_var = 0;
	for (int i = 0; i < iteration; i++) {
		double diff = (comp_rate[i] - comp_mean);
		sum_var += diff*diff;
	}
	comp_var = sum_var/iteration;
	System.out.println("For K = "+k);
	System.out.println("Compression rate mean is: "+comp_mean+", variance is: "+comp_var);

    }
    
    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k){
	int w=originalImage.getWidth();
	int h=originalImage.getHeight();
	BufferedImage kmeansImage = new BufferedImage(w,h,originalImage.getType());
	Graphics2D g = kmeansImage.createGraphics();
	g.drawImage(originalImage, 0, 0, w,h , null);
	// Read rgb values from the image
	int[] rgb=new int[w*h];
	int count=0;
	for(int i=0;i<w;i++){
	    for(int j=0;j<h;j++){
		rgb[count++]=kmeansImage.getRGB(i,j);
	    }
	}
	// Call kmeans algorithm: update the rgb values
	kmeans(rgb,k);

	// Write the new rgb values to the image
	count=0;
	for(int i=0;i<w;i++){
	    for(int j=0;j<h;j++){
		kmeansImage.setRGB(i,j,rgb[count++]);
	    }
	}
	return kmeansImage;
    }

    // Your k-means code goes here
    // Update the array rgb by assigning each entry in the rgb array to its cluster center
    private static void kmeans(int[] rgb, int k){
    	// we stores the r-g-b values for each pixel, first is red, second is green, last is blue
    	// the variantion range is 0-255
    	int total_pix = rgb.length;
    	int[] pre_cluster = new int[k];   
        int[] cluster = new int[k];   
        int[] count_cluster = new int[k];  
    	int[][] sum_cluster_rgb = new int[k][3];
        int[] assign_cluster = new int[rgb.length];
        int[][] rgbV = new int[total_pix][3];

        double maximumDistance;   
        double currentDistance = 0;                   
        int min = 0;               
        
        // pick the initial cluster centers
        for ( int i = 0; i < k; i++ ) {
			Random rand = new Random(); 
			int value = rand.nextInt(total_pix);
			cluster[i] = value;
        }
        
        // get the rgb values for each pixel
    	for (int i = 0; i < total_pix; i++) {
    		rgbV[i][0] = (rgb[i] >> 16) & 0xff; 
    		rgbV[i][1] = (rgb[i] >> 8) & 0xff;
    		rgbV[i][2] = (rgb[i]) & 0xff;
		}
     
    	// get the rgb values for the clusters
    	int[][] cluster_rgb = new int[k][3];
     	for (int i = 0; i < k; i++) {
			int value = cluster[i];
			cluster_rgb[i][0] = (value >> 16) & 0xff; 
			cluster_rgb[i][1] = (value >> 8) & 0xff;
			cluster_rgb[i][2] = (value) & 0xff;
		}   	
    	
        while( !isConverged(pre_cluster, cluster) ) {
        	
            for ( int i = 0; i < cluster.length; i++ ) {
            	pre_cluster[i] = cluster[i];
            	count_cluster[i] = 0;
            	sum_cluster_rgb[i][0] = 0;
            	sum_cluster_rgb[i][1] = 0;
            	sum_cluster_rgb[i][2] = 0;
            }

            // E-step, evaluate euci distance, assign each pixel new cluster.
            for ( int i = 0; i < rgb.length; i++ ) {
            	maximumDistance = Double.MAX_VALUE;            	
            	for ( int j = 0; j < cluster.length; j++ ) {
                    int diff_Red = rgbV[i][0]-cluster_rgb[j][0];
                    int diff_Green = rgbV[i][1]-cluster_rgb[j][1];
                    int diff_Blue = rgbV[i][2]-cluster_rgb[j][2];
//                    currentDistance = calculatePixelDistance( rgb[i], cluster[j] );
                    currentDistance = (double)Math.sqrt(Math.abs(diff_Red*diff_Red+diff_Green*diff_Green+diff_Blue*diff_Blue));
            		if ( currentDistance < maximumDistance ) {
            			maximumDistance = currentDistance;
            			min = j;
            		}
            	}
            	assign_cluster[i] = min;
            	count_cluster[min]++;
            	sum_cluster_rgb[min][0] += rgbV[i][0];
            	sum_cluster_rgb[min][1] += rgbV[i][1];
            	sum_cluster_rgb[min][2] += rgbV[i][2];
            }
            
          // M-step, update clusters
            for ( int i = 0; i < cluster.length; i++ ) {
            	int mRed = (int)((double)sum_cluster_rgb[i][0]/(double)count_cluster[i]);
            	int mGreen = (int)((double)sum_cluster_rgb[i][1]/(double)count_cluster[i]);
            	int mBlue = (int)((double)sum_cluster_rgb[i][2]/(double)count_cluster[i]);
            	cluster[i] = ((mRed&0x0ff)<<16)|((mGreen&0x0ff)<<8)|((mBlue&0x0ff));
            }
        } 
        
        // update the final pixel values 
        for ( int i = 0; i < rgb.length; i++ ) {
        	rgb[i] = cluster[assign_cluster[i]];
        }
    }

    private static boolean isConverged( int[] pre_cluster, int[] cluster ) {
        for (int i = 0; i < cluster.length; i++){
        	if (pre_cluster[i] != cluster[i]){
        		return false;
        	}        	
        }
        return true;
    } 
    private static double calculatePixelDistance( int pixelA, int pixelB ) {
        int differenceOfRed = ((pixelA & 0x00FF0000) >>> 16) - ((pixelB & 0x00FF0000) >>> 16);
        int differenceOfGreen = ((pixelA & 0x0000FF00) >>> 8)  - ((pixelB & 0x0000FF00) >>> 8);
        int differenceOfBlue = ((pixelA & 0x000000FF) >>> 0)  - ((pixelB & 0x000000FF) >>> 0);
        return Math.sqrt( differenceOfRed*differenceOfRed + differenceOfGreen*differenceOfGreen + differenceOfBlue*differenceOfBlue );
      }

}
