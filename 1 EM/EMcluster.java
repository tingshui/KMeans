import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.sql.NClob;

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

public class EMcluster {
	public static void main(String[] args) throws Exception{
		
		File file = new File(args[0]);
		int initiationType = Integer.parseInt(args[1]);
		int iteration = Integer.parseInt(args[2]);
		
		double[] data = new double[6000];
		int Ndata = 6000;
		int Ncluster = 3;
		double[] result_prob = new double[iteration]; // store results of total likelihood
		
		double[] means = new double[Ncluster];
		double[] covariance = new double[Ncluster]; // this is sigma square, not sigma
		double[] prior = new double[Ncluster];
		double[][] posterior = new double[Ndata][Ncluster];
		// these are for determining convergence
		double[] pre_means = new double[Ncluster];
		double[] pre_covariance = new double[Ncluster];
		// total likelihood
		double total_loglike = 0.0;
		
		// read in data
		BufferedReader in = new BufferedReader(new FileReader(file));
		String line = in.readLine();
		int n = 0;
		while (line != null) {
		    double datai = Double.parseDouble(line);
		    data[n] = datai;
		    n++;
		    line = in.readLine();
		}
		in.close();
		
		for (int i = 0; i < iteration; i++) {
			// Initiate variables
			initiation ini = new initiation(data, initiationType);
			ini.calculate();
			means = ini.means;
			covariance = ini.covariance;
			prior = ini.prior;		
			// EMsteps
			while(!convergence(Ncluster, pre_means, means, pre_covariance, covariance)){
				pre_means = means;
				pre_covariance = covariance;
				Estep e = new Estep(data, means, covariance, prior);
				e.calculate();
				posterior = e.posterior;
				Mstep m = new Mstep(data, posterior);
				m.calculate();
				means = m.means;
				covariance = m.covariance;
				prior = m.prior;
				total_loglike = e.total_loglike;

			}
			result_prob[i] = total_loglike;
			
			// print out results
			System.out.println("iteration "+i+":");
			for (int j = 0; j < Ncluster; j++) {				
				System.out.println("cluster "+j+": mean = "+means[j]+", covariance = "+covariance[j]);
			}		
		}	
		
		double max = result_prob[0];
		for (int i = 1; i < iteration; i++) {
			if (result_prob[i] > max){
				max = result_prob[i];
			}
		}
		System.out.println("So the maximum log likelihood is: "+max);
	}
	
	public static boolean convergence(int Ncluster, double[] pre_means, double[] means, double[] pre_covariance, double[] covariance){
		boolean converge = true;
		double thresh = 0.0001;
		for (int i = 0; i < Ncluster; i++) {
			if (Math.abs(pre_means[i] - means[i]) > thresh) {
				converge = false;
				break;
			}
			if (Math.abs(pre_covariance[i] - covariance[i]) > thresh) {
				converge = false;
				break;
			}
		}
		return converge;
	}
}
