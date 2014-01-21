package org.pMLR.ML;


import java.io.IOException;
import java.util.Vector;

import org.apache.hadoop.io.FloatWritable;
import org.pMLR.ML.LBFGS.ExceptionWithIflag;
import org.HBLR.base.Example;
import org.HBLR.base.WeightParameter;

public class IterativeMLR {
	
	public static double optimize( Vector<Example> data , WeightParameter param, float[] vparams, double lambda , double eps , int maxiter ) throws IOException {
		int m = 5, maxnfn = maxiter;
		double f = 0, xtol = 1e-30;
		boolean diagco = false;
		int[] iprint = new int[2];
		int[] iflag = new int[1];
		
		if ( vparams.length != data.size() ) {
			System.out.println(" Incorrect length of vparams and data \n\n");
			throw new IOException(" Incorrect length of vparams and data \n");
		}
		
		int n = param.weightvector.length;
		for( Example E : data )	
			n = Math.max( n , E.fsize() + 1 );			
		

		double[] x  = new double[n];
		double[] g = new double[n]; 
		double[] diag = new double[n];
		
		iprint[0] = 1000;
		iprint[1] = 0;
		iflag[0] = 0;
		
		for( int i = 0; i < param.weightvector.length; ++i )
			x[i] = param.weightvector[i];
		System.out.println("\n Training for node = " + param.node + "\n");
		LBFGS opt = new LBFGS();
		while( maxnfn > 0 ) {
			f = FunctionGradient( x , g , data , param , vparams , lambda );
			try {
				opt.lbfgs(n, m, x, f, g, diagco, diag , iprint, eps, xtol, iflag);
			} catch (ExceptionWithIflag e1) {
				e1.printStackTrace();
				System.out.print(" [ GC : LBFGS Failed : for node = " + param.node + " ] ");
				break;
			}
			if( iflag[0] <= 0 ) break;
			maxnfn--;
		}
		
		if ( maxnfn > 0 ) {
			double Gnorm = 0;
			for ( int i = 0; i < g.length; ++i )
				Gnorm += g[i]*g[i];
			Gnorm = Math.sqrt(Gnorm);
		}		
		
		// Store x into current.
		param.weightvector = new float[x.length];
		double nn = 0;
		for( int i = 0; i < x.length; ++i ) {
			param.weightvector[i] = (float) x[i];
			nn += x[i]*x[i];
		}
		System.out.print("  [sum of squares of the weights =" + nn + "] " );
		if ( Double.isNaN(nn) || Double.isInfinite(nn) || Math.abs(nn) < 1e-20 ) {
			throw new ArithmeticException("Weight vector has NaN's or Inf's or is zero !");
		}
		return f;
	}

	private static double FunctionGradient( double[] x, double[] G , Vector<Example> data , WeightParameter param ,  float[] A, double lambda ) throws ArithmeticException{
		double f = 0;
		
		G[0] = 0;
		for( int i = 1; i < x.length; ++i ) {
			G[i] = lambda * x[i];
			f += lambda/2*x[i]*x[i];
		}
		
		int cnt = 0;
		
		for( Example e : data ) {
			int y = 0;
			for ( int l : e.labels ) if ( l == param.node ) y = 1;

			double wx = x[0];  
			for ( int i = 0; i < e.fids.length; ++i )
				wx += x[ e.fids[i] ] * e.fvals[i];	
			double pr = Math.exp(wx - A[cnt]);

			// Gradient through data
			for ( int i = 0; i < e.fids.length; ++i )
				G[ e.fids[i] ] += (pr-y) * e.fvals[i];
			G[0] += 0;// (pr-y); // MUST BE ZERO ! I dont know how to handle bias term !
			
			double add = -y*wx + pr;

			if ( Double.isNaN(add) || Double.isInfinite(add) ) {
				System.out.println(" pr = " + pr + " i = " + e.docid + " y = " + y + " add = " + add );
				throw new ArithmeticException("One of the probabilities is NaN or Inf ! Is the regularization term too low ?");
			}
			f += add;
			
			cnt = cnt + 1;
		}
		return f;
	}
}
