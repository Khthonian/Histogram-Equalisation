// Calculate an intensity histogram from the input image
kernel void intHistogram(global const ushort* A, global int* B) {
	// Get the global ID of the current item and store it in a variable
	int globalID = get_global_id(0);

	// Store the value of the array at the 'ID' point and store it in a variable
	int index = A[globalID];

	// Atomically increment the value of the output array at the 'index' point
	atomic_inc(&B[index]);
}

// Calculate an intensity histogram from the input image
kernel void intHistogram2(global const ushort* A, global int* B, int imgSize, int binCount, global int* histoSizeBuffer, local int* localBuffer) {
	// Get the global ID of the current item and store it in a variable
	int globalID = get_global_id(0);

	// Get the size of all of the items and store it in a variable
	int globalSize = get_global_size(0);

	// Get the local ID of the current item and store it in a variable
	int localID = get_local_id(0);

	// Get the size of the local items and store it in a variable
	int localSize = get_local_size(0);

	// Initialise the local buffer to zero for each work item
	for (int i = localID; i < binCount; i += localSize) {
		localBuffer[i] = 0;
	}

	// Synchronise all work items in the work group
	barrier(CLK_LOCAL_MEM_FENCE);

	// Iterate over all of the pixels and increment the respective bin in the local buffer
	for (int i = globalID; i < imgSize; i += globalSize) {
		for (int j = 0; j < binCount - 1; j++) { // fixed
			if (A[i] >= histoSizeBuffer[j] && A[i] < histoSizeBuffer[j + 1]) {
				// Atomically increment the corresponding bin in the local buffer
				atomic_inc(&localBuffer[j]);
				break;
			}
		}
		if (A[i] >= histoSizeBuffer[binCount - 1]) {
			// Atomically increment the last bin in the local buffer
			atomic_inc(&localBuffer[binCount - 1]);
		}
	}

	// Synchronise all work items in the work group
	barrier(CLK_LOCAL_MEM_FENCE);

	// Add the local buffer to the global buffer to produce the final histogram
	for (int i = localID; i < binCount - 1; i += localSize) { // fixed
		// Atomically add the corresponding bin into the global buffer
		atomic_add(&B[i], localBuffer[i]);
	}
	if (localID == 0 && binCount > 0) {
		// Add the count for the last bin to the global buffer
		atomic_add(&B[binCount - 1], localBuffer[binCount - 1]);
	}
}

// Calculate a cumulative histogram
kernel void cumHistogram(global int* A, global int* B) {
	// Get the global ID of the current item and store it in a variable
	int globalID = get_global_id(0);

	// Get the size of all of the items and store it in a variable
	int globalSize = get_global_size(0);

	// Loop through the indices add the value of the current point to the current cumulative sum
	for (int i = globalID + 1; i < globalSize; i++) {
		atomic_add(&B[i], A[globalID]);
	}
}

// Calculate a cumulative histogram using the Blelloch pattern
kernel void cumHistogramB(global int* A, global int* B) {
	// Get the global ID of the current item and store it in a variable
	int globalID = get_global_id(0);

	// Get the size of all of the items and store it in a variable
	int globalSize = get_global_size(0);

	// Initialise a variable to store temporary values during swaps
	int C;

	// Iterate up through all the strides
	for (int stride = 1; stride < globalSize; stride *= 2) {
		// Check if the current item should be updated based upon the current stride and global ID
		if (((globalID + 1) % (stride * 2)) == 0) {
			A[globalID] += A[globalID - stride];
		}

		// Synchronise all work items in the work group
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	// Set the last value to 0
	if (globalID == 0) {
		A[globalSize - 1] = 0;
	}

	// Synchronise all work items in the work group
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Iterate down through all the strides
	for (int stride = globalSize / 2; stride > 0; stride /= 2) {
		// Check if the current item should be updated based upon the current stride and global ID 
		if (((globalID + 1) % (stride * 2)) == 0) {
			// Swap the values of the current item and the prior item
			C = A[globalID];
			A[globalID] += A[globalID - stride];
			A[globalID - stride] = C;
		}

		// Synchronise all work items in the work group
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	// Copy A into B
	B[globalID] = A[globalID];
}

// Calculate a cumulative histogram using the Hillis-Steel pattern
kernel void cumHistogramHS(global int* A, global int* B) {
	// Get the global ID of the current item and store it in a variable
	int globalID = get_global_id(0);

	// Get the size of all of the items and store it in a variable
	int globalSize = get_global_size(0);
	
	// Create a swap buffer
	global int* C;

	// Iterate over the strides
	for (int stride = 1; stride < globalSize; stride *= 2) {
		// Copy the value from A to B
		B[globalID] = A[globalID];
		
		// If the global ID is greater than or equal to the stride, add the value at that point
		if (globalID >= stride) {
			B[globalID] += A[globalID - stride];
		}

		// Synchronise all work items in the work group
		barrier(CLK_GLOBAL_MEM_FENCE);

		// Swap the buffers
		C = A; A = B; B = C;
	}

	// Final buffer swap
	B[globalID] = A[globalID];
	barrier(CLK_GLOBAL_MEM_FENCE);
	C = A; A = B; B = C;
}

// Calculate a cumulative histogram using the Hillis-Steele pattern
kernel void cumHistogramHS2(__global const int* A, global int* B, local int* X, local int* Y) {
	// Get the global ID of the current item and store it in a variable
	int globalID = get_global_id(0);

	// Get the local ID of the current item and store it in a variable
	int localID = get_local_id(0);

	// Get the size of the local items and store it in a variable
	int localSize = get_local_size(0);

	// Create a pointer variable
	local int* Z;

	// Copy the input data into the local memory of the work group
	X[localID] = A[globalID];

	// Synchronise all work items in the work group
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform the Hillis-Steele algorithm to calculate the cumulative histogram
	for (int i = 1; i < localSize; i *= 2) {
		// If the thread is greater than or equal to the current iterative value, add the two values from the previous iteration
		if (localID >= i) {
			Y[localID] = X[localID] + X[localID - i];
		}

		// If the thread is not greater than or equal to the current iterative value, copy the value from the input array
		else {
			Y[localID] = X[localID];
		}			

		// Synchronise all work items in the work group
		barrier(CLK_LOCAL_MEM_FENCE);

		// Swap the pointers to the input and output arrays before the next iteration
		Z = Y;
		Y = X;
		X = Z;
	}

	// Copy the output data back to the global memory
	B[globalID] = X[localID];
}

// Store the normalised cumulative histogram to a look-up table for mapping the original intensities onto the output image
kernel void lookupTable(global int* A, global int* B, const int maxIntensity) {
	// Get the global ID of the current item and store it in a variable
	int globalID = get_global_id(0);

	// Store the value of the array at the 'ID' point and store it in a variable
	int index = A[globalID];

	// Calculate the value for the output
	B[globalID] = index * (double)maxIntensity / A[maxIntensity];
}

// Store the normalised cumulative histogram to a look-up table for mapping the original intensities onto the output image
kernel void lookupTable2(global int* A, global int* B, const int maxIntensity, int binCount) {
	// Get the global ID of the current item and store it in a variable
	int globalID = get_global_id(0);

	// Store the value of the array at the 'ID' point and store it in a variable
	int index = A[globalID];

	// Normalise the histogram to a maximum, respective to the bit width.
	B[globalID] = index * (double)maxIntensity / A[binCount - 1];
}

// Back-project each output pixel by indexing the look-up table with the original intensity level
kernel void backprojection(global ushort* A, global int* LUT, global ushort* B) {
	// Get the global ID of the current item and store it in a variable
	int globalID = get_global_id(0);

	// Store the value of the array at the 'ID' point and store it in a variable
	int index = A[globalID];

	// Set the value for the output using the value from the look-up table
	B[globalID] = LUT[index];
}

// Back-project each output pixel by indexing the look-up table with the original intensity level
kernel void backprojection2(global ushort* A, global int* LUT, global ushort* B, int binCount, global int* histoSizeBuffer) {
	// Get the global ID of the current item and store it in a variable
	int globalID = get_global_id(0);

	// Store the value of the array at the 'ID' point and store it in a variable
	int index = A[globalID];

	// Loop through each bin in the histogram
	for (int i = 0; i < binCount; i++)
	{
		// Check if the input intensity falls within the current bin
		if (index >= histoSizeBuffer[i] && index < histoSizeBuffer[i + 1])
		{
			// Map the input intensity to the output intensity using the look-up table
			B[globalID] = LUT[i];
		}
	}
}