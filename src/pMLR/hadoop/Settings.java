package org.pMLR.hadoop;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;


public class Settings {
	static String dataset, select;
	/* Regularization parameter */
	static double DEFAULT_lambda = 1;
	/* solve inner iteration to accuracy eps */
	static double DEFAULT_eps = 1e-3;
	/* Max number of iterations in inner optimization */
	static int DEFAULT_innerloop_maxiter = 200;
	/* Max number of iterations in outer optimization */
	static int DEFAULT_outerloop_maxiter = 100;
	
	public static Configuration setVariables( Configuration conf , String args[] ){
		dataset = "dataset-name";		
		
		conf.setIfUnset("gc.dataset" , dataset );
		conf.setIfUnset("gc.dataset.train.loc" , "/datasets/" + dataset + "/seqfile/train/");
		conf.setIfUnset("gc.iterativemlr-train.input" , "/datasets/" + dataset + "/leaflabels/*" );
		conf.setIfUnset("gc.iterativemlr-train.output" , "/params/" + dataset + "/itermlr/weights/" );
		conf.setIfUnset("gc.iterativemlr-train-vparam.output" , "/params/" + dataset + "/itermlr/vparams/" );
		conf.setIfUnset("gc.iterativemlr-train-fvalues.dir" , "/params/" + dataset + "/itermlr/fvalues/" );
		
		conf.setIfUnset("gc.iterativemlr-train.iterations", "" + DEFAULT_outerloop_maxiter);
		conf.setIfUnset("gc.iterativemlr-train.maxiter",""+ DEFAULT_innerloop_maxiter);
		conf.setIfUnset("gc.iterativemlr-train.lambda",""+ DEFAULT_lambda);
		conf.setIfUnset("gc.iterativemlr-train.startiter", "" + 0 );		
		// The eps option is ignored, refer TrainingDriver:line 194, eps is set algorithmically instead
		conf.set("gc.iterativemlr-train.eps", "" + DEFAULT_eps );		
	
		return conf;		
	}
	
    public static Configuration addPathToDC( Configuration conf , String path ) throws IOException {
        FileSystem fs = FileSystem.get( conf );
        FileStatus[] fstatus = fs.globStatus( new Path(path) );
        Path[] listedPaths = FileUtil.stat2Paths( fstatus );
        for( Path p : listedPaths ) {   
                System.out.println(" Add File to DC " + p.toUri().toString() );
                DistributedCache.addCacheFile( p.toUri() , conf);
        }
        return conf;
    }
}