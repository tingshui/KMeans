import java.util.Random;

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
 *  reference: 
 *  1 http://www.vlfeat.org/api/gmm-fundamentals.html
 *  2 Simple Methods for Initializing the EM Algorithm for Gaussian Mixture Models
 *    Blomer and Bujna, http://arxiv.org/pdf/1312.5946.pdf
 *  
 *  1 The first method: random data points
 *  Random data points. This method sets the
 *  means of the modes by sampling at random a corresponding number of data
 *  points, sets the covariance matrices of all the modes are to the 
 *  covariance of the entire dataset, and sets the prior probabilities of
 *  the Gaussian modes to be uniform. This initialization method is the 
 *  fastest, simplest, as well as the one most likely to end in a bad
 *  local minimum.
 *  
 *  2 The second method: Kmeans initialization
 *  KMeans initialization This method uses KMeans
 *  to pre-cluster the points. It then sets the means and covariances of
 *  the Gaussian distributions the sample means and covariances of each KMeans
 *  cluster. It also sets the prior probabilities to be proportional to the 
 *  mass of each cluster. 
 *  
 *  3 random data points, variance = 1
 *  4 Kmeans initialization, variance = 1
 *---------------------------------------------------------------------*/

public class initiation {
	double[] data;
	int Ndata = 6000;
	int Ncluster = 3;
	double[] means;
	double[] covariance;
	double[] prior;
	int initiationType;

	public initiation(double[] data, int initiationType){
		this.data = data;
		this.means = new double[Ncluster];
		this.covariance = new double[Ncluster];
		this.prior = new double[Ncluster];
		this.initiationType = initiationType;
	}
	
	public void calculate(){
		if (initiationType == 1){
			exeInit1();
		}
		else if (initiationType == 2){
			exeInit2();
		}
		else if (initiationType == 3){
			exeInit3();
		}
		else{
			exeInit4();
		}
	}
	
	// The first method: random initialization. Randomly select k points in dataset as cluster center
	// Let the variance equals to the variance of the entire dataset.
	private void exeInit1(){
		double prior_v = 1. / Ncluster;		
		double mean = 0.0;
		double sum = 0.0;
		double variance = 0.0;
		double sum_variance = 0.0;
		for (int i = 0; i < data.length; i++) {
			sum += data[i];
		}
		mean = sum/(double)data.length;
		for (int i = 0; i < data.length; i++) {
			double difference = data[i] - mean;
			sum_variance += difference*difference;
		}
		variance = sum_variance/(double)data.length;		
		for (int i = 0; i < Ncluster; i++) {
			Random rand = new Random(); 
			int value = rand.nextInt(Ndata) + 1;
			means[i] = data[value];
			prior[i] = prior_v;
			covariance[i] = variance;
		}		
	}
	
	// The second method: Kmeans initialization
	private void exeInit2(){
		double[] means_ini = new double[Ncluster];
		double[] pre_means_ini = new double[Ncluster];
		double[] covariance_ini = new double[Ncluster];
		double[] pre_covariance_ini = new double[Ncluster];
		double[] prior_ini = new double[Ncluster];
		int[] assign_cluster = new int[Ndata];
		int[] count_cluster = new int[Ncluster];
		double[] sum_variance_cluster = new double[Ncluster];
		double[] sum_mean_cluster = new double[Ncluster];
		// random initialization
		for (int i = 0; i < Ncluster; i++) {
			Random rand = new Random(); 
			int value = rand.nextInt(Ndata) + 1;
			means_ini[i] = data[value];
			pre_means_ini[i] = 0.0;
			covariance_ini[i] = 1.0;
			pre_covariance_ini[i] = 0.0;
		}
		// EM step
		while(!convergence(pre_means_ini, means_ini, pre_covariance_ini, covariance_ini)){
			// again! do not directly equal two objects, even though you have initialized.
			for (int i = 0; i < Ncluster; i++) {
				pre_means_ini[i] = means_ini[i];
				pre_covariance_ini[i] = covariance_ini[i];
			}
			// assign each point
			for (int i = 0; i < data.length; i++) {
				double min = Math.abs(data[i] - means_ini[0]);
				assign_cluster[i] = 0;
				for (int j = 1; j < Ncluster; j++) {
					double difference = Math.abs(data[i] - means_ini[j]);
					if (difference < min){
						min = difference;
						assign_cluster[i] = j;
					}
				}
				count_cluster[assign_cluster[i]]++;
				double var = data[i] - means_ini[assign_cluster[i]];
				sum_variance_cluster[assign_cluster[i]] += (var*var);
				sum_mean_cluster[assign_cluster[i]] += data[i];
			}			
			// update the variance, means, and priors
			for (int i = 0; i < Ncluster; i++) {
				means_ini[i] = sum_mean_cluster[i]/count_cluster[i];
				covariance_ini[i] = sum_variance_cluster[i]/count_cluster[i];
				prior_ini[i] = count_cluster[i]/(double)Ndata;
			}			
		}
		for (int i = 0; i < Ncluster; i++) {
			means[i] = means_ini[i];
			covariance[i] = covariance_ini[i];
			prior[i] = prior_ini[i];
 		}
	}
	
	// The third method: random initialization. Randomly select k points in dataset as cluster center
	// Let the variance to be 1, every cluster have the same prior
	private void exeInit3(){
		double prior_v = 1. / Ncluster;
		for (int i = 0; i < Ncluster; i++) {
			Random rand = new Random(); 
			int value = rand.nextInt(Ndata) + 1;
			means[i] = data[value];
			prior[i] = prior_v;
			covariance[i] = 1.0;
		}		
	}

	// The forth method: Kmeans initialization
	// Let the variance to be 1, every cluster have the same prior
	private void exeInit4(){
		double[] means_ini = new double[Ncluster];
		double[] pre_means_ini = new double[Ncluster];
		double[] covariance_ini = new double[Ncluster];
		double[] pre_covariance_ini = new double[Ncluster];
		double[] prior_ini = new double[Ncluster];
		int[] assign_cluster = new int[Ndata];
		int[] count_cluster = new int[Ncluster];
		double[] sum_variance_cluster = new double[Ncluster];
		double[] sum_mean_cluster = new double[Ncluster];
		// random initialization
		for (int i = 0; i < Ncluster; i++) {
			Random rand = new Random(); 
			int value = rand.nextInt(Ndata) + 1;
			means_ini[i] = data[value];
			pre_means_ini[i] = 0.0;
			covariance_ini[i] = 1.0;
			pre_covariance_ini[i] = 0.0;
		}
		// EM step
		while(!convergence(pre_means_ini, means_ini, pre_covariance_ini, covariance_ini)){
			// again! do not directly equal two objects, even though you have initialized.
			for (int i = 0; i < Ncluster; i++) {
				pre_means_ini[i] = means_ini[i];
				pre_covariance_ini[i] = covariance_ini[i];
			}
			// assign each point
			for (int i = 0; i < data.length; i++) {
				double min = Math.abs(data[i] - means_ini[0]);
				assign_cluster[i] = 0;
				for (int j = 1; j < Ncluster; j++) {
					double difference = Math.abs(data[i] - means_ini[j]);
					if (difference < min){
						min = difference;
						assign_cluster[i] = j;
					}
				}
				count_cluster[assign_cluster[i]]++;
				double var = data[i] - means_ini[assign_cluster[i]];
				sum_variance_cluster[assign_cluster[i]] += (var*var);
				sum_mean_cluster[assign_cluster[i]] += data[i];
			}			
			// update the variance, means, and priors
			for (int i = 0; i < Ncluster; i++) {
				means_ini[i] = sum_mean_cluster[i]/count_cluster[i];
				covariance_ini[i] = sum_variance_cluster[i]/count_cluster[i];
				prior_ini[i] = count_cluster[i]/(double)Ndata;
			}			
		}
		for (int i = 0; i < Ncluster; i++) {
			means[i] = means_ini[i];
			covariance[i] = 1.0;
			prior[i] = prior_ini[i];
 		}

	}
	
	private boolean convergence(double[]pre_means_ini, double[] means_ini, double[] pre_covariance_ini, double[] covariance_ini){
		boolean converge = true;
		double thresh = 0.001;
		for (int i = 0; i < means_ini.length; i++) {
			if(Math.abs(pre_means_ini[i] - means_ini[i]) > thresh){
				converge = false;
				break;
			}
			if (Math.abs(pre_covariance_ini[i] - covariance_ini[i]) > thresh) {
				converge = false;
				break;
			}
		}
		return converge;
	}
}
