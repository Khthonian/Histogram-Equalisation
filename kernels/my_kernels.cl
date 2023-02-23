// Calculate an intensity histogram from the input image
kernel void intHistogram(global const uchar* A, global int* B) {
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
	int N = get_global_size(0);

	// Loop through the indices add the value of the current point to the current cumulative sum
	for (int i = ID + 1; i < N; i++)
		atomic_add(&B[i], A[ID]);
}

// Calculate a cumulative histogram using the Hillis-Steel pattern
kernel void cumHistogramHS(global int* A, global int* B, local int* X, local int* Y) {
	// Get the global ID of the current item and store it in a variable
	int ID = get_global_id(0);

	// Get the local ID of the current item and store it in a variable
	int localID = get_local_id(0);

	// Get the local size of all of the items and store it in a variable
	int localSize = get_local_size(0);
	
	// Create a variable for swapping the X and Y arrays
	local int* Z;

	// Read the input ID from global and store in local, for faster memory access
	X[localID] = A[ID];

	// Ensure all work-items in the work-group have finished loading into local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Iterate the cumulative histogram across the local array
	for (int i = 1; i < localSize; i *= 2) {
		// Perform the cumulative histogram using parallel reduction
		if (localID >= i)
			Y[localID] = X[localID] + X[localID - i];
		else
			Y[localID] = X[localID];

		// Ensure all work-items in the work-group have finished updating the Y array
		barrier(CLK_LOCAL_MEM_FENCE);

		// Swap the arrays around
		Z = Y;
		Y = X;
		X = Z;
	}

	// Copy the local array to the output array
	B[ID] = X[localID];
}

// Store the normalised cumulative histogram to a look-up table for mapping the original intensities onto the output image
kernel void lookupTable(global int* A, global int* B) {
	// Get the global ID of the current item and store it in a variable
	int ID = get_global_id(0);

	// Calculate the value of B[ID]
	B[ID] = A[ID] * (double)255 / A[255];
}

// Back-project each output pixel by indexing the look-up table with the original intensity level
kernel void backprojection(global uchar* A, global int* LUT, global uchar* B) {
	// Get the global ID of the current item and store it in a variable
	int ID = get_global_id(0);

	// Set the value of B[ID] using the value from the look-up table
	B[ID] = LUT[A[ID]];
}