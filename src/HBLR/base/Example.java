package org.HBLR.base;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;


public class Example implements Writable {
	public int[] fids;
	public float[] fvals;
	public int[] labels;
	public int docid;

	final void normalize(){
		double tot = 0;
		for( float fv : fvals )
			tot += fv * fv;
		tot = Math.sqrt( tot );
		if( tot > 0 ) {
			for(int i =0; i < fvals.length; ++i )  
				fvals[i] /= tot;
		}
	}
	
	public int fsize(){
		if( fids.length > 0 )
			return fids[ fids.length - 1 ];
		return 0;
	}
	
	public Example( final Example e ){
		this.docid = e.docid;
		this.labels = e.labels.clone();
		this.fids = e.fids.clone();
		this.fvals = e.fvals.clone();
	}

	public Example(){
		fids = new int[0];
		fvals = new float[0];
		labels = new int[0];
		docid = 0;
	}

	public Example(Text value) {
		fids = new int[0];
		fvals = new float[0];
		labels = new int[0];
		parseString( value.toString() );
	}		

	public Example( String x ){
		fids = new int[0];
		fvals = new float[0];
		labels = new int[0];
		docid = 0;
		parseString( x );
	}

	private void parseString( String s ){	
		s = s.replace(':', ' ');
		StringTokenizer stok = new StringTokenizer(s);
		int nf = (stok.countTokens()-1-1-1)/2; // -1 for the label, -1 for #, -1 for docid

		fids = new int[nf];
		fvals = new float[nf];
		StringTokenizer stok_labels = new StringTokenizer( stok.nextToken().replace(',', ' ') );

		int nl = stok_labels.countTokens();
		labels = new int[nl]; 
		for( int i = 0; i < nl; ++i ) labels[i] = Integer.parseInt( stok_labels.nextToken() );


		for( int cnt = 0; ; ++cnt ){
			String str_a = stok.nextToken();
			String str_b = stok.nextToken();

			int a = -1;
			if( str_a.charAt(0) != '#' ) a = Integer.parseInt( str_a );
			float b = Float.parseFloat( str_b );
			if( a == -1 ){
				docid = (int) b;
				break;
			} else {
				fids[cnt] = a;
				fvals[cnt] = b;				
			}
		}
	}

	public void print(){
		System.out.print( labels.toString() );
		for ( int i = 0; i < fids.length; ++i )
			System.out.print("  " + fids[i] + ":" + fvals[i] );
		System.out.println(" # " + docid + "\n");
	}


	public void readFields(DataInput in) throws IOException {
		fids = new int[0];
		fvals = new float[0];
		labels = new int[0];
		
		int nl = in.readInt();
		labels = new int[nl];
		for ( int i = 0; i < nl; ++i ) {
			labels[i] = in.readInt();
		}
		int nf = in.readInt();
		fids = new int[nf];
		fvals = new float[nf];
		
		for( int i = 0; i < nf; ++i ){
			int a = in.readInt();
			float b = in.readFloat();
			fids[i] = a;
			fvals[i] = b;				
		}
		docid = in.readInt();
	}

	public void write(DataOutput out) throws IOException {
		out.writeInt( labels.length );
		for ( int i = 0; i < labels.length; ++i ) 
			out.writeInt( labels[i] );
		
		out.writeInt( fids.length );
		for ( int i = 0; i < fids.length; ++i ) {
			out.writeInt( fids[i] );
			out.writeFloat( fvals[i] );
		}
		out.writeInt( docid );			
	}

	public String toString() {
		String str = "" + labels.toString();
		for( int i = 0; i < fids.length; ++i ){
			str = str + " " + fids[i] + ":" + fvals[i];
		}
		str = str + " # " + docid;
		return str;
	}
}