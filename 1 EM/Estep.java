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

public class Estep {
	double[] data;
	int Ndata;
	int Ncluster;
	double[] means;
	double[] covariance;
	double[] prior;
	double[][] posterior;
	double total_loglike = 0.0;


	public Estep(double[] data, double[] means, double[] covariance, double[] prior){
		Ndata = data.length;
		Ncluster = means.length;
		this.data = data;
		this.means = means;
		this.covariance = covariance;
		this.prior = prior;
		this.posterior = new double[Ndata][Ncluster];
	}
	
	public void calculate(){

		for(int i = 0; i< Ndata; i++){
			double total_class_j =0;
			for(int j = 0; j < Ncluster; j++){
				posterior[i][j] = prior[j] * Gaussian(data[i], means[j], covariance[j]);
				total_class_j += posterior[i][j]; 
			}
			total_loglike += Math.log(total_class_j);
			// Normalization for each data item
			for(int k = 0; k < Ncluster; k++){
				posterior[i][k] = posterior[i][k]/total_class_j;
			}
		}
		
		total_loglike = total_loglike/(double)Ndata;
	}
	
	private double Gaussian(double x, double mean, double covariance){
		double prob = 0.0;
	    prob = Math.pow(2.* Math.PI * covariance, -1/2.) * Math.exp(-(Math.pow(x - mean, 2))/(2 * covariance));
	    return prob;
	}
}
