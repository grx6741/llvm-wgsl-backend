let device = null;
let shaderCode = null;

// Load shader from file
async function loadShader() {
	const response = await fetch('axpy_output.wgsl');
	if (!response.ok) {
		throw new Error(`Failed to load shader: ${response.statusText}`);
	}
	return await response.text();
}

// Initialize WebGPU
async function initWebGPU() {
	if (device) return device;

	if (!navigator.gpu) {
		throw new Error("WebGPU not supported");
	}

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		throw new Error("No GPU adapter found");
	}

	device = await adapter.requestDevice();
	return device;
}

// Show status message
function showStatus(message, type = 'loading') {
	const statusDiv = document.getElementById('status');
	statusDiv.textContent = message;
	statusDiv.className = `status ${type}`;
	statusDiv.style.display = 'block';
}

// Run AXPY kernel with detailed timing
async function runAXPY(n, a, x_data, y_data) {
	const timings = {};

	// Timing: Load shader
	let t0 = performance.now();
	if (!shaderCode) {
		shaderCode = await loadShader();
	}
	timings.shaderLoad = performance.now() - t0;

	// Timing: GPU initialization
	t0 = performance.now();
	const device = await initWebGPU();
	timings.gpuInit = performance.now() - t0;

	// Timing: Create shader module
	t0 = performance.now();
	const shaderModule = device.createShaderModule({
		code: shaderCode
	});
	timings.shaderCompile = performance.now() - t0;

	// Timing: Create buffers and upload data
	t0 = performance.now();

	const x_buffer = device.createBuffer({
		size: x_data.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		mappedAtCreation: true
	});
	new Float32Array(x_buffer.getMappedRange()).set(x_data);
	x_buffer.unmap();

	const y_buffer = device.createBuffer({
		size: y_data.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
		mappedAtCreation: true
	});
	new Float32Array(y_buffer.getMappedRange()).set(y_data);
	y_buffer.unmap();

	const uniformData = new ArrayBuffer(8);
	const uniformView = new DataView(uniformData);
	uniformView.setFloat32(0, a, true);
	uniformView.setInt32(4, n, true);

	const uniformBuffer = device.createBuffer({
		size: uniformData.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		mappedAtCreation: true
	});
	new Uint8Array(uniformBuffer.getMappedRange()).set(new Uint8Array(uniformData));
	uniformBuffer.unmap();

	timings.bufferCreation = performance.now() - t0;

	// Timing: Create pipeline
	t0 = performance.now();

	const bindGroupLayout = device.createBindGroupLayout({
		entries: [
			{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
			{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
			{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
		]
	});

	const bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [
			{ binding: 0, resource: { buffer: x_buffer } },
			{ binding: 1, resource: { buffer: y_buffer } },
			{ binding: 2, resource: { buffer: uniformBuffer } }
		]
	});

	const pipeline = device.createComputePipeline({
		layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
		compute: { module: shaderModule, entryPoint: 'wgsl_main' }
	});

	timings.pipelineCreation = performance.now() - t0;

	// Timing: Encode and submit commands
	t0 = performance.now();

	const commandEncoder = device.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	const workgroupSize = 256;
	const numWorkgroups = Math.ceil(n / workgroupSize);
	passEncoder.dispatchWorkgroups(numWorkgroups, 1, 1);

	passEncoder.end();

	const readBuffer = device.createBuffer({
		size: y_data.byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	commandEncoder.copyBufferToBuffer(y_buffer, 0, readBuffer, 0, y_data.byteLength);

	const gpuCommands = commandEncoder.finish();

	timings.commandEncoding = performance.now() - t0;

	// Timing: GPU execution (submit + wait)
	t0 = performance.now();
	device.queue.submit([gpuCommands]);
	await device.queue.onSubmittedWorkDone();
	timings.gpuExecution = performance.now() - t0;

	// Timing: Read results
	t0 = performance.now();
	await readBuffer.mapAsync(GPUMapMode.READ);
	const resultArray = new Float32Array(readBuffer.getMappedRange()).slice();
	readBuffer.unmap();
	timings.resultReadback = performance.now() - t0;

	// Cleanup
	x_buffer.destroy();
	y_buffer.destroy();
	uniformBuffer.destroy();
	readBuffer.destroy();

	// Total time
	timings.total = Object.values(timings).reduce((a, b) => a + b, 0);

	return { result: resultArray, timings, numWorkgroups };
}

// Display metrics
function displayMetrics(timings, n, numWorkgroups) {
	const metricsDiv = document.getElementById('metrics');
	metricsDiv.innerHTML = `
				<div class="metric">
					<span class="metric-label">Shader Load:</span>
					<span class="metric-value">${timings.shaderLoad.toFixed(2)} ms</span>
				</div>
				<div class="metric">
					<span class="metric-label">GPU Init:</span>
					<span class="metric-value">${timings.gpuInit.toFixed(2)} ms</span>
				</div>
				<div class="metric">
					<span class="metric-label">Shader Compile:</span>
					<span class="metric-value">${timings.shaderCompile.toFixed(2)} ms</span>
				</div>
				<div class="metric">
					<span class="metric-label">Buffer Creation:</span>
					<span class="metric-value">${timings.bufferCreation.toFixed(2)} ms</span>
				</div>
				<div class="metric">
					<span class="metric-label">Pipeline Creation:</span>
					<span class="metric-value">${timings.pipelineCreation.toFixed(2)} ms</span>
				</div>
				<div class="metric">
					<span class="metric-label">Command Encoding:</span>
					<span class="metric-value">${timings.commandEncoding.toFixed(2)} ms</span>
				</div>
				<div class="metric">
					<span class="metric-label">🔥 GPU Execution:</span>
					<span class="metric-value">${timings.gpuExecution.toFixed(2)} ms</span>
				</div>
				<div class="metric">
					<span class="metric-label">Result Readback:</span>
					<span class="metric-value">${timings.resultReadback.toFixed(2)} ms</span>
				</div>
				<div class="metric" style="border-top: 2px solid #4fc3f7; margin-top: 10px; padding-top: 10px;">
					<span class="metric-label"><strong>Total Time:</strong></span>
					<span class="metric-value">${timings.total.toFixed(2)} ms</span>
				</div>
			`;

	const throughputDiv = document.getElementById('throughput');
	const dataSize = n * 4 * 3; // 3 arrays (x, y, result) * 4 bytes per float32
	const dataSizeMB = dataSize / (1024 * 1024);
	const bandwidth = dataSizeMB / (timings.gpuExecution / 1000); // MB/s
	const gflops = (n * 2) / (timings.gpuExecution * 1e6); // 2 FLOPs per element (mul + add)

	throughputDiv.innerHTML = `
				<div class="metric">
					<span class="metric-label">Array Size:</span>
					<span class="metric-value">${n.toLocaleString()} elements</span>
				</div>
				<div class="metric">
					<span class="metric-label">Data Size:</span>
					<span class="metric-value">${dataSizeMB.toFixed(2)} MB</span>
				</div>
				<div class="metric">
					<span class="metric-label">Workgroups:</span>
					<span class="metric-value">${numWorkgroups}</span>
				</div>
				<div class="metric">
					<span class="metric-label">Bandwidth:</span>
					<span class="metric-value">${bandwidth.toFixed(2)} MB/s</span>
				</div>
				<div class="metric">
					<span class="metric-label">Performance:</span>
					<span class="metric-value">${gflops.toFixed(2)} GFLOPS</span>
				</div>
			`;
}

// Validate results
function validateResults(x, y_original, y_result, a, n) {
	const samplesToShow = 10;
	let errors = 0;
	let maxError = 0;

	let output = `Checking first ${samplesToShow} elements:\n\n`;
	output += `Expected: y[i] = a*x[i] + y_original[i] = ${a}*x[i] + y_original[i]\n\n`;

	for (let i = 0; i < Math.min(samplesToShow, n); i++) {
		const expected = a * x[i] + y_original[i];
		const actual = y_result[i];
		const error = Math.abs(expected - actual);

		maxError = Math.max(maxError, error);
		if (error > 1e-5) errors++;

		output += `[${i}] x=${x[i].toFixed(2)}, y_orig=${y_original[i].toFixed(2)} => `;
		output += `expected=${expected.toFixed(2)}, got=${actual.toFixed(2)}`;
		output += error > 1e-5 ? `(error: ${error.toFixed(6)})\n` : ` ✓\n`;
	}

	output += `\nMax error: ${maxError.toExponential(2)}\n`;
	output += errors === 0 ? `\nAll samples correct!` : `\n${errors} errors found`;

	document.getElementById('validation').textContent = output;
}

// Run single benchmark
async function runBenchmark() {
	try {
		const n = parseInt(document.getElementById('arraySize').value);
		const a = parseFloat(document.getElementById('scalarValue').value);

		showStatus('Initializing...', 'loading');

		// Initialize arrays
		const x = new Float32Array(n);
		const y_original = new Float32Array(n);

		for (let i = 0; i < n; i++) {
			x[i] = i * 1.0;
			y_original[i] = i * 2.0;
		}

		showStatus('Running kernel...', 'loading');

		const { result, timings, numWorkgroups } = await runAXPY(n, a, x, y_original.slice());

		displayMetrics(timings, n, numWorkgroups);
		validateResults(x, y_original, result, a, n);

		document.getElementById('shaderInfo').textContent =
			`Entry Point: wgsl_main\n` +
			`Workgroup Size: 256x1x1\n` +
			`Num Workgroups: ${numWorkgroups}x1x1\n` +
			`Total Threads: ${numWorkgroups * 256}\n` +
			`Shader Lines: ${shaderCode.split('\n').length}`;

		showStatus('Benchmark completed successfully!', 'success');

	} catch (error) {
		showStatus(`Error: ${error.message}`, 'error');
		console.error(error);
	}
}

// Run multiple benchmarks and average
async function runMultipleBenchmarks() {
	try {
		const n = parseInt(document.getElementById('arraySize').value);
		const a = parseFloat(document.getElementById('scalarValue').value);
		const runs = 50;

		showStatus(`Running ${runs} benchmarks...', 'loading`);

		const x = new Float32Array(n);
		const y_original = new Float32Array(n);

		for (let i = 0; i < n; i++) {
			x[i] = i * 1.0;
			y_original[i] = i * 2.0;
		}

		const allTimings = [];

		for (let i = 0; i < runs; i++) {
			showStatus(`Running benchmark ${i + 1}/${runs}...`, 'loading');
			const { timings, numWorkgroups } = await runAXPY(n, a, x, y_original.slice());
			allTimings.push(timings);

			// Small delay between runs
			await new Promise(resolve => setTimeout(resolve, 100));
		}

		// Calculate averages
		const avgTimings = {};
		for (const key in allTimings[0]) {
			avgTimings[key] = allTimings.reduce((sum, t) => sum + t[key], 0) / runs;
		}

		displayMetrics(avgTimings, n, Math.ceil(n / 256));

		showStatus(`Completed ${runs} runs (showing averages)`, 'success');

	} catch (error) {
		showStatus(`Error: ${error.message}`, 'error');
		console.error(error);
	}
}

// Auto-run on page load
window.addEventListener('load', () => {
	showStatus('Ready. Click "Run Benchmark" to start.', 'success');
});
