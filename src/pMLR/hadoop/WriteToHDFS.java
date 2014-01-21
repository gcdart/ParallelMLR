package org.pMLR.hadoop;

import java.io.FileInputStream;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class WriteToHDFS {
	static void write( Configuration conf , String fval ,  String contents ) throws IOException{
		FileSystem fileSystem = FileSystem.get(conf);

		// Check if the file already exists
		Path path = new Path(fval);
		if (fileSystem.exists(path)) {
			System.out.println("File " + fval + " already exists");
			return;
		}

		// Create a new file and write data to it.
		FSDataOutputStream out = fileSystem.create(path);
		out.write( contents.getBytes() );
		out.close();
	}
}
