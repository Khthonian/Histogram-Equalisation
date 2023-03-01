// Calculate an intensity histogram from the input image
kernel void intHistogram(global const ushort* A, global int* B) {
	// Get the global ID of the current item and store it in a variable
	int ID = get_global_id(0);

	// Store the value of the array at the 'ID' point and store it in a variable
	int index = A[ID];

	// Increment the value of the output array at the 'index' point
	atomic_inc(&B[index]);
}

kernel void intHistogramB(global const ushort* A, global int* H, local int* LH, int A_size, int histBins, global int* binsizeBuffer) {
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int lsize = get_local_size(0);
	int gsize = get_global_size(0);

	// Set Local Histogram Bins to 0
	for (int i = lid; i < histBins; i += lsize)
	{
		LH[i] = 0;
	}

	// Wait for all threads to finish setting local histogram bins to 0
	barrier(CLK_LOCAL_MEM_FENCE);

	// Compute Local Histogram
	for (int i = gid; i < A_size; i += gsize)
	{
		for (size_t j = 0; j < histBins; j++)
		{
			if (A[i] >= binsizeBuffer[j] && A[i] < binsizeBuffer[j + 1])
			{
				atomic_inc(&LH[j]);
				break;
			}
		}
	}

	// Wait for all threads to finish computing local histogram
	barrier(CLK_LOCAL_MEM_FENCE);

	// Copy Local Histograms to Global Histogram
	for (int i = lid; i < histBins; i += lsize)
	{
		atomic_add(&H[i], LH[i]);
	}
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

// Calculate a cumulative histogram using the Hillis-Steel pattern
kernel void cumHistogramHS2(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int* scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
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