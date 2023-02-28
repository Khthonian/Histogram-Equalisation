// Calculate an intensity histogram from the input image
kernel void intHistogram(global const ushort* A, global int* B) {
	// Get the global ID of the current item and store it in a variable
	int ID = get_global_id(0);

	//Store the value of the array at the 'ID' point and store it in a variable
	int index = A[ID];

	// Increment the value of the output array at the 'index' point
	atomic_inc(&B[index]);
}

// Calculate a cumulative histogram
kernel void cumHistogram(global int* A, global int* B) {
	// Get the global ID of the current item and store it in a variable
	int ID = get_global_id(0);

	// Get the size of all of the items and store it in a variable
	int X = get_global_size(0);

	// Loop through the indices add the value of the current point to the current cumulative sum
	for (int i = ID + 1; i < X; i++)
		atomic_add(&B[i], A[ID]);
}

// Calculate a cumulative histogram using the Hillis-Steel pattern
kernel void cumHistogramHS(global int* A, global int* B) {
	// Get the global ID of the current item and store it in a variable
	int ID = get_global_id(0);

	// Get the local size of all of the items and store it in a variable
	int size = get_global_size(0);
	
	// Create a swap buffer
	global int* C;

	for (int stride = 1; stride < size; stride *= 2) {
		B[ID] = A[ID];
		if (ID >= stride)
			B[ID] += A[ID - stride];

		barrier(CLK_GLOBAL_MEM_FENCE);

		C = A;
		A = B;
		B = C;
	}
}

// Store the normalised cumulative histogram to a look-up table for mapping the original intensities onto the output image
kernel void lookupTable(global int* A, global int* B, const int maxIntensity) {
	// Get the global ID of the current item and store it in a variable
	int ID = get_global_id(0);

	// Calculate the value of B[ID]
	B[ID] = A[ID] * (double)maxIntensity / A[maxIntensity];
}

// Back-project each output pixel by indexing the look-up table with the original intensity level
kernel void backprojection(global ushort* A, global int* LUT, global ushort* B) {
	// Get the global ID of the current item and store it in a variable
	int ID = get_global_id(0);

	// Set the value of B[ID] using the value from the look-up table
	B[ID] = LUT[A[ID]];
}