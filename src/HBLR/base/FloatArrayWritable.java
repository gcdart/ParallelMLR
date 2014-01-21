package org.HBLR.base;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;

public class FloatArrayWritable extends ArrayWritable
{
    public FloatArrayWritable() {
        super(FloatWritable.class);
    }
    public FloatArrayWritable(FloatWritable[] values) {
        super(FloatWritable.class, values);
    }
    
    public String toString(){
    	double ret = 0;
    	Writable[] ww = get();
    	for ( Writable w : ww ){
    		ret += ((FloatWritable) w).get();
    	}
    	return "" + ret;
    }
}