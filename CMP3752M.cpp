/*
Submission
- Stewart Charles Fisher II - ID: 25020928
- Parallel Programming - CMP3752M
- Assignment 1
*/

/*
Description
- The code is designed to handle 8-bit and 16-bit imagery, of both greyscale and RGB varieties.
- The contents of this code contain adaptations and improvements of Tutorial 2 and Tutorial 3, for the base code and the kernel functions. This can be found at https://github.com/wing8/OpenCL-Tutorials
- There is also a kernel function that has been adapted from https://github.com/spoolean/HistogramEqualisation
- The images that the code was tested on include .ppm and .pgm images. These images can be found in the relevant directories.
- The intensity histogram implementations feature a serial implementation and a parallel reduction implementation.
- The cumulative histogram implementations feature a simple implementation, two variations of the Hillis-Steele pattern, and a single implementation of the Blelloch pattern.
- The user is able to give their own desired bin count, which can affect the output of the image and the histograms produced.
- The user is also able to select the functions being used.
- Performance metrics and the histograms are displayed to the user via the console.
- Each step of the model will be indicated as follows: "STEP X - XXXXX"
*/

#include <iostream>
#include <vector>
#include "include/Utils.h"
#include "include/CImg.h"

using namespace cimg_library;

// Define a type specifically for the image
typedef unsigned short modularImage;

// A function to display instructions for using the program
void printHelp() {
	std::cerr << "Application usage:" << std::endl;

	// Prompt to select a platform
	std::cerr << "  -p : select platform (Default: 0)" << std::endl;

	// Prompt to select a device
	std::cerr << "  -d : select device (Default: 0)" << std::endl;

	// Prompt to list all devices and platforms
	std::cerr << "  -l : list all platforms and devices" << std::endl;

	// Prompt to input an image file
	std::cerr << "  -f : input image file (Default: test.pgm)" << std::endl;

	// Prompt to display the instructions again
	std::cerr << "  -h : print this message" << std::endl;
}

void printProfiling(string step, string kernelFunctionName, cl::Event kernelEvent, vector<int> kernelValues = {}) {
	// Calculate and print the intensity histogram kernel execution time
	std::cout << std::endl << step << " Kernel Function: " << kernelFunctionName << std::endl;

	std::cout << std::endl << step << " Kernel Execution Time [ns]: " << kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

	std::cout << std::endl << step << " Kernel Memory Transfer: " << GetFullProfilingInfo(kernelEvent, ProfilingResolution::PROF_NS) << std::endl;

	if (!kernelValues.empty()) {
		std::cout << std::endl << step << " Values:" << std::endl << kernelValues << std::endl;
	}

	std::cout << std::endl << "--------------------------------------------------" << std::endl;
}

// A function to display the output image, varied by bit depth
CImgDisplay displayOutputImage(CImg<modularImage> image, bool is16BitUsed) {
	// Check the bit depth
	if (is16BitUsed) {
		CImg<unsigned short> newImage = (CImg<modularImage>) image;
		CImgDisplay displayOutputImage(newImage, "Output");
		return displayOutputImage;
	}

	else {
		CImg<unsigned char> newImage = (CImg<modularImage>) image;
		CImgDisplay displayOutput(newImage, "Output");
		return displayOutput;
	}
}

int main(int argc, char** argv) {
	// Set the default platform and device to 0
	int platformID = 0;
	int deviceID = 0;

	// Set the default image file to test.pgm
	string imgFile = "test.pgm";

	// Iterate through the command line arguments
	for (int i = 1; i < argc; i++) {
		// Set the platform ID as the selected platform
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platformID = atoi(argv[++i]); }

		// Set the device ID as the selected device
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { deviceID = atoi(argv[++i]); }

		// List all devices and platforms
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }

		// Set the image file name as the selected image file
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { imgFile = argv[++i]; }

		// Display the instructions and terminate the program
		else if (strcmp(argv[i], "-h") == 0) { printHelp(); return 0; }
	}

	// Disable CImg library exception handling
	cimg::exception_mode(0);

	// A variable to hold the input of the user for the bin count
	string userInput;

	// A variable to hold the converted user input
	int binCount = 0;

	// A variable to hold the input of the user for the intensity histogram
	int intHistoChoice;

	// A variable to hold the input of the user for the intensity histogram
	int cumHistoChoice;

	// A variable to hold the input of the user for the intensity histogram
	int lookupChoice;

	// A variable to hold the input of the user for the intensity histogram
	int backprojectChoice;

	// A variable to store whether a 16-bit image was used
	bool is16BitUsed;

	// A variable to store whether an RGB image was used
	bool rgbUsed;

	// A variable to store the max intensity of the look-up table
	int maxIntensity = 255;

	// A variable to store the console output for 8 and 16-bit
	int consoleVariant = 256;

	// Try to apply the histogram equalisation algorithm
	try {
		/*
		STEP 1 ---------------- IMAGE PREPARATION ----------------
		*/

		// Open the image file
		CImg<unsigned char> displayImgInput(imgFile.c_str());

		CImg<unsigned short> tempImgInput(imgFile.c_str());

		std::cout << "Loaded image is " << imgFile << std::endl;

		// Check if the image is 16-bit
		if (tempImgInput.max() <= 255) {
			std::cout << "Loaded image is 8-bit." << std::endl;
			is16BitUsed = false;
			maxIntensity = 255;
			consoleVariant = 256;
		}

		else if (tempImgInput.max() <= 65535) {
			std::cout << "Loaded image is 16-bit." << std::endl;
			is16BitUsed = true;
			maxIntensity = 65535;
			consoleVariant = 65536;
		}

		// RGB to YCbCr Conversion
		CImg<unsigned short> imgInput;
		CImg<unsigned short> cbChannel, crChannel;

		if (tempImgInput.spectrum() == 1) {
			std::cout << "Loaded image is greyscale." << std::endl;
			imgInput = tempImgInput;
			rgbUsed = false;
		}
		else if (tempImgInput.spectrum() == 3) {
			std::cout << "Loaded image is RGB." << std::endl;
			rgbUsed = true;

			// Convert RGB image to YCbCr
			CImg<unsigned short> ycbcrImage = tempImgInput.get_RGBtoYCbCr();

			// Extract the channels setting y as the input image
			imgInput = ycbcrImage.get_channel(0);
			cbChannel = ycbcrImage.get_channel(1);
			crChannel = ycbcrImage.get_channel(2);
		}

		/*
		STEP 2 ---------------- MODEL SELECTION ----------------
		*/

		// Prompt to enter a bin count
		std::cout << "Enter a bin count between 1 and 256: " << "\n";

		// Loop until a valid input has been received
		while (true)
		{
			// Store user input in the pre-made variable
			getline(std::cin, userInput);

			// Check if the user input is an empty string and prompt the user to enter a valid input
			if (userInput == "") { std::cout << "Please enter a number." << "\n"; continue; }

			// Try to convert the user input to an integer and store it in the pre-made variable
			try { binCount = std::stoi(userInput); }

			// If the user input is not an integer, catch the exception and prompt the user for a valid input
			catch (...) { std::cout << "Please enter an integer." << "\n"; continue; }

			// Check if the user input is in the range of 1 and the maximum, and exit with the break statement
			if (binCount >= 1 && binCount <= 256) { break; }

			// If the user input is not within the valid range, prompt the user to enter a valid input
			else { std::cout << "Please enter a number in between 1 and 256: " << "\n"; continue; }
		}

		// Prompt to enter a selection for the intensity histogram
		std::cout << "\n" << "Enter an option for the intensity histogram: " << "\n";
		std::cout << "1) Serial Implementation" << "\n";
		std::cout << "2) Local Memory Implementation" << "\n";

		// Loop until a valid input has been received
		while (true)
		{
			// Store user input in the pre-made variable
			getline(std::cin, userInput);

			// Check if the user input is an empty string and prompt the user to enter a valid input
			if (userInput == "") { std::cout << "Please enter a number." << "\n"; continue; }

			// Try to convert the user input to an integer and store it in the pre-made variable
			try { intHistoChoice = std::stoi(userInput); }

			// If the user input is not an integer, catch the exception and prompt the user for a valid input
			catch (...) { std::cout << "Please enter an integer." << "\n"; continue; }

			// Check if the user input is in the range of 1 and the maximum, and exit with the break statement
			if (intHistoChoice >= 1 && intHistoChoice <= 2) { break; }

			// If the user input is not within the valid range, prompt the user to enter a valid input
			else { std::cout << "Please enter 1 or 2: " << "\n"; continue; }
		}

		// Switch the kernel according to choice.
		string intHistoFunction;
		switch (intHistoChoice) {
			case 1:
				intHistoFunction = "intHistogram";
				break;
			case 2:
				intHistoFunction = "intHistogram2";
				break;
		}

		// Prompt to enter a selection for the cumulative histogram
		std::cout << "\n" << "Enter an option for the cumulative histogram: " << "\n";
		std::cout << "1) Serial Implementation" << "\n";
		std::cout << "2) Blelloch Implementation" << "\n";
		std::cout << "3) Hillis-Steele Implementation" << "\n";
		std::cout << "4) Double Buffered Hillis-Steele Implementation" << "\n";

		// Loop until a valid input has been received
		while (true)
		{
			// Store user input in the pre-made variable
			getline(std::cin, userInput);

			// Check if the user input is an empty string and prompt the user to enter a valid input
			if (userInput == "") { std::cout << "Please enter a number." << "\n"; continue; }

			// Try to convert the user input to an integer and store it in the pre-made variable
			try { cumHistoChoice = std::stoi(userInput); }

			// If the user input is not an integer, catch the exception and prompt the user for a valid input
			catch (...) { std::cout << "Please enter an integer." << "\n"; continue; }

			// Check if the user input is in the range of 1 and the maximum, and exit with the break statement
			if (cumHistoChoice >= 1 && cumHistoChoice <= 4) { break; }

			// If the user input is not within the valid range, prompt the user to enter a valid input
			else { std::cout << "Please enter a number between 1 and 4: " << "\n"; continue; }
		}

		// Switch the kernel according to choice.
		string cumHistoFunction;
		switch (cumHistoChoice) {
			case 1:
				cumHistoFunction = "cumHistogram";
				break;
			case 2:
				cumHistoFunction = "cumHistogramB";
				break;
			case 3:
				cumHistoFunction = "cumHistogramHS";
				break;
			case 4:
				cumHistoFunction = "cumHistogramHS2";
				break;
		}

		// Prompt to enter a selection for the look-up table
		std::cout << "\n" << "Enter an option for the look-up table: " << "\n";
		std::cout << "1) Standardised Implementation" << "\n";
		std::cout << "2) Variable Implementation" << "\n";

		// Loop until a valid input has been received
		while (true)
		{
			// Store user input in the pre-made variable
			getline(std::cin, userInput);

			// Check if the user input is an empty string and prompt the user to enter a valid input
			if (userInput == "") { std::cout << "Please enter a number." << "\n"; continue; }

			// Try to convert the user input to an integer and store it in the pre-made variable
			try { lookupChoice = std::stoi(userInput); }

			// If the user input is not an integer, catch the exception and prompt the user for a valid input
			catch (...) { std::cout << "Please enter an integer." << "\n"; continue; }

			// Check if the user input is in the range of 1 and the maximum, and exit with the break statement
			if (lookupChoice >= 1 && lookupChoice <= 2) { break; }

			// If the user input is not within the valid range, prompt the user to enter a valid input
			else { std::cout << "Please enter 1 or 2: " << "\n"; continue; }
		}

		// Switch the kernel according to choice.
		string lookupFunction;
		switch (lookupChoice) {
		case 1:
			lookupFunction = "lookupTable";
			break;
		case 2:
			lookupFunction = "lookupTable2";
			break;
		}

		// Prompt to enter a selection for the back-projection
		std::cout << "\n" << "Enter an option for the back-projection: " << "\n";
		std::cout << "1) Standardised Implementation" << "\n";
		std::cout << "2) Variable Implementation" << "\n";

		// Loop until a valid input has been received
		while (true)
		{
			// Store user input in the pre-made variable
			getline(std::cin, userInput);

			// Check if the user input is an empty string and prompt the user to enter a valid input
			if (userInput == "") { std::cout << "Please enter a number." << "\n"; continue; }

			// Try to convert the user input to an integer and store it in the pre-made variable
			try { backprojectChoice = std::stoi(userInput); }

			// If the user input is not an integer, catch the exception and prompt the user for a valid input
			catch (...) { std::cout << "Please enter an integer." << "\n"; continue; }

			// Check if the user input is in the range of 1 and the maximum, and exit with the break statement
			if (backprojectChoice >= 1 && backprojectChoice <= 2) { break; }

			// If the user input is not within the valid range, prompt the user to enter a valid input
			else { std::cout << "Please enter 1 or 2: " << "\n"; continue; }
		}

		// Switch the kernel according to choice.
		string backprojectFunction;
		switch (backprojectChoice) {
		case 1:
			backprojectFunction = "backprojection";
			break;
		case 2:
			backprojectFunction = "backprojection2";
			break;
		}

		/*
		STEP 3 ---------------- MODEL PREPARATION ----------------
		*/

		// Display the original input image
		CImgDisplay displayInput(displayImgInput, "input");

		// Create an OpenCL context object, with the platform and device to be used
		cl::Context context = GetContext(platformID, deviceID);

		// Print the platform ID and device ID being used
		std::cout << "\n" << "Running on " << GetPlatformName(platformID) << ", " << GetDeviceName(platformID, deviceID) << std::endl;

		// Enable profiling for the command, to measure the program performance
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Set up the sources for the OpenCL program and add the kernel file, which contains the necessary functions
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");

		// Create the OpenCL program from the sources
		cl::Program program(context, sources);

		// Try to build the OpenCL program
		try {
			program.build();
		}

		// If there are errors building the program, output the status, options, and log to the console, and throw the error
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		/*
		STEP 4 ---------------- BUFFER PREPARATION ----------------
		*/

		// Create a vector for the intensity histogram with the size of the user-defined bin count
		std::vector<int> IH(binCount);

		// Create a vector to determine the size of the increments for the histogram, based upon the bin count
		std::vector<int> binValues(binCount);
		int increments = consoleVariant / binCount;
		for (int i = 0; i < binCount; i++)
		{
			binValues[i] = i * increments;
		}

		// Calculate the total size of the histogram in bytes
		size_t histoSize = binCount * sizeof(int);

		// Create an OpenCL buffer for the input image
		cl::Buffer imgInputBuffer(context, CL_MEM_READ_ONLY, imgInput.size() * sizeof(imgInput[0]));

		// Create an OpenCL buffer for the output image
		cl::Buffer imgOutputBuffer(context, CL_MEM_READ_WRITE, imgInput.size() * sizeof(imgInput[0]));

		// Create an OpenCL buffer for the intensity histogram
		cl::Buffer intHistoBuffer(context, CL_MEM_READ_WRITE, histoSize);

		// Create an OpenCL buffer for the cumulative histogram
		cl::Buffer cumHistoBuffer(context, CL_MEM_READ_WRITE, histoSize);

		// Create an OpenCL buffer for the look-up table
		cl::Buffer lookupBuffer(context, CL_MEM_READ_WRITE, histoSize);

		// Create an OpenCL buffer equal to the histogram size
		cl::Buffer histoSizeBuffer(context, CL_MEM_READ_WRITE, histoSize);

		/*
		STEP 5 ---------------- INTENSITY HISTOGRAM ----------------
		*/

		// Write the input image data to the relevant device buffer
		queue.enqueueWriteBuffer(imgInputBuffer, CL_TRUE, 0, imgInput.size() * sizeof(imgInput[0]), &imgInput.data()[0]);

		queue.enqueueWriteBuffer(histoSizeBuffer, CL_TRUE, 0, histoSize, &binValues[0]);

		queue.enqueueFillBuffer(intHistoBuffer, 0, 0, histoSize);

		cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();

		std::cout << std::endl;
		std::cout << "Max work-group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
		std::cout << "Max work-item dimensions: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
		std::cout << "Max work-item sizes: ";
		std::vector<size_t> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
		for (size_t i = 0; i < maxWorkItemSizes.size(); ++i) {
			std::cout << maxWorkItemSizes[i] << " ";
		}
		std::cout << std::endl;
		std::cout << "Local memory size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;

		// Switch the kernel according to choice.
		switch (intHistoChoice) {
			case 1:

				break;
		}
		
		// Prepare the kernel for the intensity histogram
		cl::Kernel intHistoKernel = cl::Kernel(program, intHistoFunction.c_str());

		// Switch the kernel according to choice.
		switch (intHistoChoice) {
			case 1:
				// Set the arguments for the intensity histogram
				intHistoKernel.setArg(0, imgInputBuffer);
				intHistoKernel.setArg(1, intHistoBuffer);
				break;
			case 2:
				// Set the arguments for the intensity histogram
				intHistoKernel.setArg(0, imgInputBuffer);
				intHistoKernel.setArg(1, intHistoBuffer);
				intHistoKernel.setArg(2, (int)imgInput.size());
				intHistoKernel.setArg(3, binCount);
				intHistoKernel.setArg(4, histoSizeBuffer);
				intHistoKernel.setArg(5, cl::Local(histoSize));
				break;
		}

		// Run the intensity histogram event on the device
		cl::Event intHistoEvent;
		queue.enqueueNDRangeKernel(intHistoKernel, cl::NullRange, cl::NDRange(imgInput.size()), cl::NullRange, NULL, &intHistoEvent);
		
		// Read the intensity histogram data from the device back to the host
		queue.enqueueReadBuffer(intHistoBuffer, CL_TRUE, 0, histoSize, &IH[0]);

		/*
		STEP 6 ---------------- CUMULATIVE HISTOGRAM ----------------
		*/

		// Create a vector for the cumulative histogram with the size of the user-defined bin count
		std::vector<int> CH(binCount);

		// Fill the cumulative histogram buffer with zeros
		queue.enqueueFillBuffer(cumHistoBuffer, 0, 0, histoSize);

		// Prepare the kernel for the cumulative histogram	
		cl::Kernel cumHistoKernel = cl::Kernel(program, cumHistoFunction.c_str());

		// Switch the kernel according to choice.
		switch (cumHistoChoice) {
		case 1:
		case 2:
		case 3:
			// Set the arguments for the cumulative histogram
			cumHistoKernel.setArg(0, intHistoBuffer);
			cumHistoKernel.setArg(1, cumHistoBuffer);
			break;
		case 4:
			// Set the arguments for the cumulative histogram
			cumHistoKernel.setArg(0, intHistoBuffer);
			cumHistoKernel.setArg(1, cumHistoBuffer);
			cumHistoKernel.setArg(2, cl::Local(histoSize));
			cumHistoKernel.setArg(3, cl::Local(histoSize));
			break;
		}

		// Run the cumulative histogram event on the device
		cl::Event cumHistoEvent;
		queue.enqueueNDRangeKernel(cumHistoKernel, cl::NullRange, cl::NDRange(IH.size()), cl::NullRange, NULL, &cumHistoEvent);

		// Read the cumulative histogram data from the device back to the host
		queue.enqueueReadBuffer(cumHistoBuffer, CL_TRUE, 0, histoSize, &CH[0]);

		/*
		STEP 7 ---------------- LOOK-UP TABLE ----------------
		*/

		// Create a vector for the look-up-table with the size of the user-defined bin count
		std::vector<int> LUT(binCount);

		// Fill the look-up table buffer with zeros
		queue.enqueueFillBuffer(lookupBuffer, 0, 0, histoSize);

		// Prepare the kernel for the look-up table
		cl::Kernel lookupKernel = cl::Kernel(program, lookupFunction.c_str());

		// Switch the kernel according to choice.
		switch (lookupChoice) {
		case 1:
			// Set the arguments for the look-up table
			lookupKernel.setArg(0, cumHistoBuffer);
			lookupKernel.setArg(1, lookupBuffer);
			lookupKernel.setArg(2, maxIntensity);
			break;
		case 2:
			// Set the arguments for the look-up table
			lookupKernel.setArg(0, cumHistoBuffer);
			lookupKernel.setArg(1, lookupBuffer);
			lookupKernel.setArg(2, maxIntensity);
			lookupKernel.setArg(3, binCount);
			break;
		}

		// Run the look-up table event
		cl::Event lookupEvent;
		queue.enqueueNDRangeKernel(lookupKernel, cl::NullRange, cl::NDRange(IH.size()), cl::NullRange, NULL, &lookupEvent);

		// Read the look-up table data from the device back to the host
		queue.enqueueReadBuffer(lookupBuffer, CL_TRUE, 0, histoSize, &LUT[0]);

		/*
		STEP 8 ---------------- BACK-PROJECTION ----------------
		*/

		// Prepare the kernel for the back-projection
		cl::Kernel backprojectKernel = cl::Kernel(program, backprojectFunction.c_str());

		// Switch the kernel according to choice.
		switch (backprojectChoice) {
		case 1:
			// Set the arguments for the back-projection
			backprojectKernel.setArg(0, imgInputBuffer);
			backprojectKernel.setArg(1, lookupBuffer);
			backprojectKernel.setArg(2, imgOutputBuffer);
			break;
		case 2:
			// Set the arguments for the back-projection
			backprojectKernel.setArg(0, imgInputBuffer);
			backprojectKernel.setArg(1, lookupBuffer);
			backprojectKernel.setArg(2, imgOutputBuffer);
			backprojectKernel.setArg(3, binCount);
			backprojectKernel.setArg(4, histoSizeBuffer);
			break;
		}

		// Run the back-projection event
		cl::Event backprojectEvent;

		/*
		STEP 9 ---------------- MODEL OUTPUT AND PERFORMANCE ----------------
		*/

		// Create a vector for the output image data with the size of the input image
		vector<unsigned short> outputData(imgInput.size());
		queue.enqueueNDRangeKernel(backprojectKernel, cl::NullRange, cl::NDRange(imgInput.size()), cl::NullRange, NULL, &backprojectEvent);

		// Read the output image data from the device back to the host
		queue.enqueueReadBuffer(imgOutputBuffer, CL_TRUE, 0, imgInput.size() * sizeof(imgInput[0]), &outputData.data()[0]);

		// Print the profiling values
		printProfiling("Intensity Histogram", intHistoFunction, intHistoEvent, IH);

		printProfiling("Cumulative Histogram", cumHistoFunction, cumHistoEvent, CH);

		printProfiling("Look-up Table", lookupFunction, lookupEvent, LUT);

		printProfiling("Back-Projection", backprojectFunction, backprojectEvent);

		// Calculate and print the total execution time of the kernels
		std::cout << std::endl << "Total Kernel Execution Time [ns]: " << backprojectEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - intHistoEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		CImg<modularImage> imgOutput(outputData.data(), imgInput.width(), imgInput.height(), tempImgInput.depth(), imgInput.spectrum());

		// Check if the image used RGB 
		if (rgbUsed == true) {
			// Create a new image with the same width and height as the initial input image, with three colour channels
			CImg<unsigned short> outputYCbCr = imgOutput.get_resize(tempImgInput.width(), tempImgInput.height(), 1, 3);

			// Loop through each pixel in the image
			for (int x = 0; x < outputYCbCr.width(); x++) {
				for (int y = 0; y < outputYCbCr.height(); y++) {
					// Set the first channel of the image to the equalised values
					outputYCbCr(x, y, 0) = imgOutput(x, y);
					// Set the second channel of the new image to the chroma blue channel values of the initial input image
					outputYCbCr(x, y, 1) = cbChannel(x, y);
					// Set the third channel of the new image to the chroma red channel values of the initial input image
					outputYCbCr(x, y, 2) = crChannel(x, y);
				}
			}

			// Convert the image back into RGB
			imgOutput = outputYCbCr.get_YCbCrtoRGB();
		}

		// Display the final equalised image
		CImgDisplay displayOutput = displayOutputImage(imgOutput, is16BitUsed);

		// Close the input image and output image windows if the ESC key is pressed
		while (!displayInput.is_closed() && !displayOutput.is_closed()
			&& !displayInput.is_keyESC() && !displayOutput.is_keyESC()) {
			displayInput.wait(1);
			displayOutput.wait(1);
		}
	}

	// Catch exception errors and print the error
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}