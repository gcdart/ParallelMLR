package org.pMLR.hadoop;
import java.io.IOException;
import java.util.Vector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;
import org.HBLR.base.Example;
import org.HBLR.base.FloatArrayWritable;


public class getFromDC {

	public static Vector<Example> getData( Configuration conf , String regex ) throws IOException{
		System.out.println(" Opening Instances = " );            

		Vector<Example> data = new Vector<Example>();
		Path[] listedPaths = DistributedCache.getLocalCacheFiles( conf );
		FileSystem fs = FileSystem.getLocal( conf );	
		
		// Parse the training data
		for( Path p : listedPaths ) if( p.toString().indexOf(regex) >= 0 ) {
			System.out.println(" Opening data file " + p.toString() );
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,p,conf);
			Writable Key = (Writable) ReflectionUtils.newInstance(reader.getKeyClass(), conf);
			Writable Value = (Writable) ReflectionUtils.newInstance(reader.getValueClass(), conf);
			
			int cnt = 0;
			while( reader.next(Key,Value) ){
				data.add( new Example((Example)Value) );
				cnt++;
				if ( cnt%50000 == 0 ) {
					Runtime r = Runtime.getRuntime();
					r.gc();
					double mb = (1024*1024);
					System.out.println(" \tSetup Memory ");
					System.out.println(" \tUsed Memory:" + (r.totalMemory() - r.freeMemory())/mb );
					System.out.println(" \tTotal Memory:" + r.totalMemory()/mb );
					System.out.println(" \tFree Memory:" + r.freeMemory()/mb );
					System.out.println(" \tMax Memory:" + (r.maxMemory())/mb );
					System.out.println( " \tdata.size() = " + data.size() );
				}
			}
		}
		System.out.println(" Loaded " + data.size()+ " Instances ");
		return data;
	}

	public static Writable[] getVariationalParams( Configuration conf , String regex ) throws IOException{
		System.out.println(" Opening Instances = " );            

		Vector<Example> data = new Vector<Example>();
		Path[] listedPaths = DistributedCache.getLocalCacheFiles( conf );
		FileSystem fs = FileSystem.getLocal( conf );	
		
		// Parse the training data
		for( Path p : listedPaths ) if( p.toString().indexOf(regex) >= 0 ) {
			System.out.println(" Opening data file " + p.toString() );
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,p,conf);
			Writable Key = (Writable) ReflectionUtils.newInstance(reader.getKeyClass(), conf);
			Writable Value = (Writable) ReflectionUtils.newInstance(reader.getValueClass(), conf);
			reader.next(Key,Value);
			FloatArrayWritable array = (FloatArrayWritable) Value;
			return array.get();
		}
		System.err.println(" RETURNING NULL !!! ");
		return null;
	}
}