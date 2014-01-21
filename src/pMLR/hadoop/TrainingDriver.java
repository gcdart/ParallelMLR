package org.pMLR.hadoop;
import java.io.IOException;
import java.util.Vector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.HBLR.base.Example;
import org.HBLR.base.FloatArrayWritable;
import org.HBLR.base.WeightParameter;

public class TrainingDriver extends Configured implements Tool {
	
	/* Input : Class-labels in each line 
	 * Output : Zero weight-vectors
	 */
	public static class IterativeMLRFirstMapper extends Mapper<LongWritable,Text,IntWritable,WeightParameter>  {
		public void map(LongWritable key, Text v, Context context ) throws IOException, InterruptedException {
			int node = Integer.parseInt(v.toString());
			WeightParameter w = new WeightParameter();
			w.node = node;
			context.write( new IntWritable(node) , w );
			System.out.println(" Writing node = " + node );
		}
	}
	
	/* Input : Weight-vectors
	 * Output : Optimized weight vectors
	 * Distributed Cache: Training data & Variational Parameters
	 */
	public static class IterativeMLRMapper extends Mapper<IntWritable,WeightParameter,IntWritable,WeightParameter> {
		double eps = Settings.DEFAULT_eps, lambda = Settings.DEFAULT_lambda;
		int maxiter = Settings.DEFAULT_innerloop_maxiter;
		Vector<Example> data = null;
		float[] vparams = null;
		
		protected void setup(Context context ) throws IOException {
			Configuration conf = context.getConfiguration();
			System.out.println(" Opening Training Instances = " 
					+ context.getConfiguration().get("gc.dataset.train.loc") );  
			/* These values are typically set before */
			conf.setIfUnset("gc.iterativemlr-train.lambda",""+lambda);
			conf.setIfUnset("gc.iterativemlr-train.eps",""+eps);
			conf.setIfUnset("gc.iterativemlr-train.maxiter",""+maxiter);
			
			lambda = Double.parseDouble(conf.get("gc.iterativemlr-train.lambda"));			
			eps = Double.parseDouble(conf.get("gc.iterativemlr-train.eps"));
			maxiter = Integer.parseInt(conf.get("gc.iterativemlr-train.maxiter"));
			
			data = getFromDC.getData( conf , conf.get("gc.dataset.train.loc") );
			Writable[] temp = getFromDC.getVariationalParams( conf , conf.get("gc.iterativemlr-train-vparam.output") );
			vparams = new float[temp.length];
			for ( int i = 0; i < temp.length; ++i ) 
				vparams[i] = ((FloatWritable)temp[i]).get();
		}
		
		public void map(IntWritable key, WeightParameter param, Context context ) throws IOException, InterruptedException {
			int node = param.node;
			
			double optx = org.pMLR.ML.IterativeMLR.optimize(data, param, vparams,lambda,eps,maxiter);

			double nn = 0;
			for ( int i = 0; i < param.weightvector.length; ++i ) 
				nn += param.weightvector[i] * param.weightvector[i];
			nn = Math.sqrt(nn);
			System.out.println("\n\t [Norm = " + nn + "]");
			
			context.write( new IntWritable(node) , param );		
			
			
			Runtime r = Runtime.getRuntime();
			r.gc();
			double mb = (1024*1024);
			System.out.println(" \tTotal Memory:" + r.totalMemory()/mb );
			System.out.println(" \tMax Memory:" + (r.maxMemory())/mb );
			
			System.out.println(" F-value = " + optx );
			
			/* In case you want to write out the F-value to a file */
			//String fval_fname = context.getConfiguration().get("gc.iterativemlr-train-fvalues.iter.dir") + node + ".value";
			//WriteToHDFS.write( context.getConfiguration() , fval_fname , "" + optx + "\n" );
		}
	}
	
	/* Input : Weight-vectors
	 * Output : partial sums of variational parameters
	 * Distributed Cache: Training data
	 */
	public static class VParamsMapper extends Mapper<IntWritable,WeightParameter,IntWritable,FloatArrayWritable>  {
		float[] sums = null;
		Vector<Example> data = null;
		
		protected void setup(Context context ) throws IOException {
			System.out.println(" Opening Training Instances = "	+ context.getConfiguration().get("gc.dataset.train.loc") );            
			data = getFromDC.getData( context.getConfiguration() , "train" );
			sums = new float[data.size()];
		}
		
		public void map(IntWritable key, WeightParameter param, Context context ) {
			for ( int i = 0; i < data.size(); ++i ) {
				double wx = param.getScore( data.get(i) );
				sums[i] += Math.exp(wx);
			}
		}
		
		protected void cleanup(Context context ) throws IOException, InterruptedException {
			FloatWritable[] writableSums = new FloatWritable[sums.length];
			for ( int i = 0; i < sums.length; ++i ) writableSums[i] = new FloatWritable( sums[i] );	
			context.write( new IntWritable(1) ,  new FloatArrayWritable(writableSums) );
		}
	}

	/* Input : partial sums of variational parameters
	 * Output: partial sums of variational parameters
	 */
	public static class VParamsCombiner extends Reducer<IntWritable,FloatArrayWritable,IntWritable,FloatArrayWritable>  {
		public void reduce( IntWritable key , Iterable<FloatArrayWritable> values , Context context ) throws IOException , InterruptedException {
			double[] sums = null;
			for ( FloatArrayWritable fl : values ) {
				Writable[] flarray = (Writable[]) fl.get();
				
				if ( sums == null ) sums = new double[flarray.length];
				if ( sums.length != flarray.length ) {
					System.out.println(" INCORRECT SIZE FOUND \n");
					throw new IOException("inconsistent sizes of vparam sizes");
				}
				
				for ( int i = 0; i < flarray.length; ++i ) sums[i] += ((FloatWritable)flarray[i]).get();
			}
			FloatWritable[] output = new FloatWritable[sums.length];
			for ( int i = 0; i < sums.length; ++i ) 
				output[i] = new FloatWritable( (float) sums[i] );
			context.write(key, new FloatArrayWritable(output) );
		}
	}
	
	/* Input : partial sums of variational parameters
	 * Output : variational parameters
	 */
	public static class VParamsReducer extends Reducer<IntWritable,FloatArrayWritable,IntWritable,FloatArrayWritable>  {
		public void reduce( IntWritable key , Iterable<FloatArrayWritable> values , Context context ) throws IOException , InterruptedException {
			double[] sums = null;
			for ( FloatArrayWritable fl : values ) {
				Writable[] flarray = (Writable[]) fl.get();
				
				if ( sums == null ) sums = new double[flarray.length];
				if ( sums.length != flarray.length ) {
					System.out.println(" INCORRECT SIZE FOUND \n");
					throw new IOException("inconsistent sizes of vparam sizes");
				}
				
				for ( int i = 0; i < flarray.length; ++i ) sums[i] += ((FloatWritable)flarray[i]).get();
			}
			for ( int i = 0; i < sums.length; ++i ) sums[i] = Math.log(sums[i]);
			
			FloatWritable[] output = new FloatWritable[sums.length];
			for ( int i = 0; i < sums.length; ++i ) 
				output[i] = new FloatWritable( (float) sums[i] );
			context.write(key, new FloatArrayWritable(output) );
		}
	}
	
	
    public int run(String[] args) throws Exception {
    	Configuration conf = new Configuration(getConf());
    	conf = Settings.setVariables( conf , args );
    	
        String input, output , vinput , voutput;
        
        int iterations = conf.getInt("gc.iterativemlr-train.iterations",Settings.DEFAULT_outerloop_maxiter);
        int start = conf.getInt("gc.iterativemlr-train.startiter", 0 );
        
        for ( int i = start; i < iterations; ++i ) {
        	Configuration j1conf = new Configuration(conf);
        	j1conf.setIfUnset("gc.iterativemlr-train-fvalues.iter.dir",
        			j1conf.get("gc.iterativemlr-train-fvalues.dir") + i +"/");
        	
        	/* Change the epsilon, i.e. make it more stricter progressively */
        	double eps = Math.max( Math.pow(10,-i/10) , 1e-5 );
        	j1conf.set("gc.iterativemlr-train.eps", "" + eps );
        	System.out.println( " Training with eps = " + eps );
        	
        	
        	if ( i == 0 ) {
        		 input = j1conf.get("gc.iterativemlr-train.input");
        	     output = j1conf.get("gc.iterativemlr-train.output") + i + "/";
        	     j1conf.set("mapred.max.split.size", j1conf.get("gc.iterativemlr-train.split","64000000"));
        	}
        	else {
        		input = j1conf.get("gc.iterativemlr-train.output") + (i-1) + "/";
        		output = j1conf.get("gc.iterativemlr-train.output") + i + "/";
        	    j1conf = Settings.addPathToDC(j1conf, j1conf.get("gc.dataset.train.loc" )  + "*");
        		j1conf = Settings.addPathToDC(j1conf, j1conf.get("gc.iterativemlr-train-vparam.output") + (i-1) + "/*" );
        	}        	
        	
        	j1conf.set("mapred.compress.map.output","true");
        	
        	Job job1 = new Job( j1conf );
        	job1.setJarByClass( TrainingDriver.class );
        	job1.setJobName( "Iterative-MLR-train-"+ "-" + i );

        	if ( i == 0 ) {
        		job1.setMapperClass( IterativeMLRFirstMapper.class);
        		job1.setInputFormatClass( TextInputFormat.class );
        	}
        	else {
        		job1.setMapperClass( IterativeMLRMapper.class );
        		job1.setInputFormatClass( SequenceFileInputFormat.class );
        	}
        	
        	job1.setMapOutputKeyClass( IntWritable.class );
        	job1.setMapOutputValueClass( WeightParameter.class );

        	job1.setOutputFormatClass( SequenceFileOutputFormat.class );
        	job1.setOutputKeyClass( IntWritable.class );
        	job1.setOutputValueClass( WeightParameter.class );        	
        	SequenceFileOutputFormat.setCompressOutput(job1, true);
        	
        	job1.setNumReduceTasks(0);
        	
        	FileInputFormat.setInputPaths( job1 , input );
        	FileOutputFormat.setOutputPath(job1 , new Path(output) );

        	System.out.println(" Input dir = " + input );
        	System.out.println(" Output dir = " + output );
        	System.out.println(" Training Input = " + conf.get("gc.dataset.train.loc" ) );
        	System.out.println(" Name = " + " Iterative-MLR-train-"+ "-" + i );
                
        	if( job1.waitForCompletion(true) == false ) {
        		System.err.println(" Job  Failed (miserably)");
        		System.exit(2);
        	}
        	
        	Configuration j2conf = new Configuration(conf);
    	    j2conf = Settings.addPathToDC(j2conf, j2conf.get("gc.dataset.train.loc" )  + "*");
    	    
        	vinput = output;
        	voutput = conf.get("gc.iterativemlr-train-vparam.output") + i + "/";
        	Job job2 = new Job( j2conf );
        	job2.setJarByClass( TrainingDriver.class );
        	job2.setJobName( "Iterative-MLR-vparam-train-"+ "-" + i );

        	job2.setMapperClass( VParamsMapper.class );
        	job2.setInputFormatClass( SequenceFileInputFormat.class );
        	
        	job2.setMapOutputKeyClass( IntWritable.class );
        	job2.setMapOutputValueClass( FloatArrayWritable.class );

        	job2.setOutputFormatClass( SequenceFileOutputFormat.class );
        	job2.setOutputKeyClass( IntWritable.class );
        	job2.setOutputValueClass( FloatArrayWritable.class );
        	
        	job2.setCombinerClass( VParamsCombiner.class );
        	job2.setReducerClass( VParamsReducer.class );
        	SequenceFileOutputFormat.setCompressOutput(job2, true);
        	
        	job2.setNumReduceTasks(1);
        	FileInputFormat.setInputPaths( job2 , vinput );
        	FileOutputFormat.setOutputPath( job2 , new Path(voutput) );

        	System.out.println(" Input dir = " + vinput );
        	System.out.println(" Output dir = " + voutput );
        	System.out.println(" Name = " + " Iterative-MLR-vparams-train-"+ "-" + i );
                
        	if( job2.waitForCompletion(true) == false ) {
        		System.err.println(" Job  Failed (miserably)");
        		System.exit(2);
        	}        	
        }
        
        return 0;
    }
    
    public static void main( String args[] ) throws Exception {
        ToolRunner.run( new Configuration(), new TrainingDriver() , args );
    }
} 	
