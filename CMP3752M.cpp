#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

// A function to display instructions for using the program
void print_help() {
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
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	// Disable CImg library exception handling
	cimg::exception_mode(0);

	// A variable to hold the input of the user
	string userInput;

	// A variable to hold the converted user input
	int binCount = 0;

	// Prompt to enter a bin count
	std::cout << "Enter a bin count in between 1 and 256." << "\n";
	
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

		// Check if the user input is in the range of 1 and 256 and exit with the break statement
		if (binCount >= 1 && binCount <= 256) { break; }

		// If the user input is not within the valid range, prompt the user to enter a valid input
		else { std::cout << "Please enter a number in between 1 and 256." << "\n"; continue; }
	}

	// Try to apply the histogram equalisation algorithm
	try {
		// Open the image file
		CImg<unsigned char> imgInput(imgFile.c_str());

		// Display the original input image
		CImgDisplay displayInput(imgInput, "input");

		// Create an OpenCL context object, with the platform and device to be used
		cl::Context context = GetContext(platformID, deviceID);

		// Print the platform ID and device ID being used
		std::cout << "Running on " << GetPlatformName(platformID) << ", " << GetDeviceName(platformID, deviceID) << std::endl;

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

		// Create a vector for the intensity histogram with the size of the user-defined bin count
		std::vector<int> IH(binCount);

		// Calculate the total size of the histogram in bytes
		size_t histoSize = IH.size() * sizeof(int);

		// Create an OpenCL buffer for the input image
		cl::Buffer imgInputBuffer(context, CL_MEM_READ_ONLY, imgInput.size());

		// Create an OpenCL buffer for the output image
		cl::Buffer imgOutputBuffer(context, CL_MEM_READ_WRITE, imgInput.size());

		// Create an OpenCL buffer for the intensity histogram
		cl::Buffer intHistoBuffer(context, CL_MEM_READ_WRITE, histoSize);

		// Create an OpenCL buffer for the cumulative histogram
		cl::Buffer cumHistoBuffer(context, CL_MEM_READ_WRITE, histoSize);

		// Create an OpenCL buffer for the look-up table
		cl::Buffer lookupBuffer(context, CL_MEM_READ_WRITE, histoSize);

		// Write the input image data to the relevant device buffer
		queue.enqueueWriteBuffer(imgInputBuffer, CL_TRUE, 0, imgInput.size(), &imgInput.data()[0]);

		// Prepare the kernel for the intensity histogram
		cl::Kernel intHistoKernel = cl::Kernel(program, "intHistogram");

		// Set the arguments for the intensity histogram
		intHistoKernel.setArg(0, imgInputBuffer);
		intHistoKernel.setArg(1, intHistoBuffer);

		// Run the intensity histogram event on the device
		cl::Event intHistoEvent; 
		queue.enqueueNDRangeKernel(intHistoKernel, cl::NullRange, cl::NDRange(imgInput.size()), cl::NullRange, NULL, &intHistoEvent);

		// Read the intensity histogram data from the device back to the host
		queue.enqueueReadBuffer(intHistoBuffer, CL_TRUE, 0, histoSize, &IH[0]);

		// Create a vector for the cumulative histogram with the size of the user-defined bin count
		std::vector<int> CH(binCount);

		// Fill the cumulative histogram buffer with zeros
		queue.enqueueFillBuffer(cumHistoBuffer, 0, 0, histoSize);

		// Prepare the kernel for the cumulative histogram
		cl::Kernel cumHistoKernel = cl::Kernel(program, "cumHistogram");

		// Set the arguments for the cumulative histogram
		cumHistoKernel.setArg(0, intHistoBuffer);
		cumHistoKernel.setArg(1, cumHistoBuffer);

		// Run the cumulative histogram event on the device
		cl::Event cumHistoEvent;
		queue.enqueueNDRangeKernel(cumHistoKernel, cl::NullRange, cl::NDRange(histoSize), cl::NullRange, NULL, &cumHistoEvent);

		// Read the cumulative histogram data from the device back to the host
		queue.enqueueReadBuffer(cumHistoBuffer, CL_TRUE, 0, histoSize, &CH[0]);

		// Create a vector for the look-up-table with the size of the user-defined bin count
		std::vector<int> LUT(binCount);

		// Fill the look-up table buffer with zeros
		queue.enqueueFillBuffer(lookupBuffer, 0, 0, histoSize);

		// Prepare the kernel for the look-up table
		cl::Kernel lookupKernel = cl::Kernel(program, "lookupTable");

		// Set the arguments for the look-up table
		lookupKernel.setArg(0, cumHistoBuffer);
		lookupKernel.setArg(1, lookupBuffer);

		// Run the look-up table event
		cl::Event lookupEvent;
		queue.enqueueNDRangeKernel(lookupKernel, cl::NullRange, cl::NDRange(histoSize), cl::NullRange, NULL, &lookupEvent);

		// Read the look-up table data from the device back to the host
		queue.enqueueReadBuffer(lookupBuffer, CL_TRUE, 0, histoSize, &LUT[0]);

		// Prepare the kernel for the back-projection
		cl::Kernel backprojectKernel = cl::Kernel(program, "backprojection");

		// Set the arguments for the back-projection
		backprojectKernel.setArg(0, imgInputBuffer);
		backprojectKernel.setArg(1, lookupBuffer);
		backprojectKernel.setArg(2, imgOutputBuffer);

		// Run the back-projection event
		cl::Event backprojectEvent;

		// Create a vector for the output image data with the size of the input image
		vector<unsigned char> outputData(imgInput.size());
		queue.enqueueNDRangeKernel(backprojectKernel, cl::NullRange, cl::NDRange(imgInput.size()), cl::NullRange, NULL, &backprojectEvent);
		
		// Read the output image data from the device back to the host
		queue.enqueueReadBuffer(imgOutputBuffer, CL_TRUE, 0, outputData.size(), &outputData.data()[0]);

		// Print the intensity histogram values
		std::cout << std::endl << "Histogram Values: " << IH << std::endl;

		// Calculate and print the intensity histogram kernel execution time
		std::cout << std::endl << "Histogram Kernel Execution Time [ns]: " << intHistoEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - intHistoEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		// Print the cumulative histogram values
		std::cout << std::endl << "Cumulative Histogram Values: " << CH << std::endl;

		// Calculate and print the cumulative histogram kernel execution time
		std::cout << std::endl << "Cumulative Histogram Kernel Execution Time [ns]: " << cumHistoEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumHistoEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		// Print the look-up table values
		std::cout << std::endl << "Look-Up Table (LUT) Values: " << LUT << std::endl;

		// Calculate and print the look-up table kernel execution time
		std::cout << std::endl << "LUT Kernel Execution Time [ns]: " << lookupEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lookupEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		// Calculate and print the back-projection kernel execution time
		std::cout << std::endl << "Vector Kernel Execution Time [ns]: " << backprojectEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - backprojectEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		// Create and display the output image from the output image buffer
		CImg<unsigned char> outputImage(outputData.data(), imgInput.width(), imgInput.height(), imgInput.depth(), imgInput.spectrum());
		CImgDisplay displayOutput(outputImage, "output");

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