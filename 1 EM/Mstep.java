/*----------------------------------------------------------------------
 *  Author: Qianying Lin
 *  Written: 4/1/2016
 *  Last updated: 4/1/2016
 * 
 *  Complication: 
 *  Execution: 
 *
 *  This file perform the EM algorithm to cluster the unlabeled data into
 *  three categories.
 *---------------------------------------------------------------------*/

public class Mstep {
	double[] data;
	int Ndata;
	int Ncluster;
	double[] means;
	double[] covariance;
	double[] prior;
	double[][] posterior;
	double total_loglike;

	public Mstep(double[] data, double[][] posterior){
		Ndata = data.length;
		Ncluster = posterior[0].length;
		this.data = data;
		this.means = new double[Ncluster];
		this.covariance = new double[Ncluster];
		this.prior = new double[Ncluster];
		this.posterior = posterior;
		this.total_loglike = 0.0;
	}
	
	public void calculate(){
		
		for( int i = 0; i < Ncluster; i++ ) {
	        double sum = 0.0;
	        for(int j = 0; j < Ndata; j++) {
	        	sum += posterior[j][i];
	        }
	        prior[i] = sum / Ndata; // update prior
	      
	        double sum_post = 0.0;
	        for(int k = 0; k < Ndata; k++) {
	        	sum_post += (posterior[k][i] * data[k]); // update means

	        }
	        means[i] = sum_post / sum;
	        
	        double sum_post_variance = 0.0;
	        for(int l = 0; l < Ndata; l++) {
	        	sum_post_variance += (posterior[l][i] * Math.pow((data[l]-means[i]), 2)); 
	        }
	        covariance[i] = sum_post_variance / sum; // update covariance	      
	    }		
	}
}
