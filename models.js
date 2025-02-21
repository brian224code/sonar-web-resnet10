import { addLog } from './util.js'
import * as tf from '@tensorflow/tfjs'

class Model {
	constructor() {
	}

	summary() {
		this.model.summary()
	}

	forward(x, shape) {
		const tensor = (x instanceof tf.Tensor) ? x : tf.tensor2d([x], shape)
		const output = this.model.predict(tensor)
		addLog('Output:', output.arraySync())
		return output
	}
}

// resnet
export class ResNet10 extends Model {
	constructor() {
		super()
		addLog("Initializing ResNet10 instance...")
		this.model = this.buildModel()
	}

	// Build the model
	buildModel() {
		const inputs = tf.input({ shape: [2352] });

		let x = tf.layers.reshape({ targetShape: [28, 28, 3] }).apply(inputs);

		// Initial Conv Layer
		x = tf.layers.conv2d({
			filters: 64,
			kernelSize: 3,
			strides: 1,
			padding: 'same',
			useBias: false
		}).apply(x);

		x = tf.layers.batchNormalization().apply(x);
		x = tf.layers.reLU().apply(x);

		// Residual Blocks
		x = this.residualBlock(x, 64);
		x = this.residualBlock(x, 128, true);
		x = this.residualBlock(x, 256, true);
		x = this.residualBlock(x, 512, true);

		// Global Average Pooling
		x = tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }).apply(x);

		// Fully Connected Layer
		x = tf.layers.dense({ units: 8, activation: 'softmax' }).apply(x);

		const model = tf.model({ inputs, outputs: x });

		model.compile({
			optimizer: 'adam',
			loss: 'categoricalCrossentropy',
			metrics: ['accuracy']
		})

		addLog('model initialized.')

		return model;
	}

	// Function to create a residual block
	residualBlock(x, filters, downsample = false) {
		let shortcut = x;

		if (downsample) {
			shortcut = tf.layers.conv2d({
				filters: filters,
				kernelSize: 1,
				strides: 2,
				padding: 'same',
				useBias: false
			}).apply(shortcut);
			
			shortcut = tf.layers.batchNormalization().apply(shortcut);
		}

		let out = tf.layers.conv2d({
			filters: filters,
			kernelSize: 3,
			strides: downsample ? 2 : 1,
			padding: 'same',
			useBias: false
		}).apply(x);
		
		out = tf.layers.batchNormalization().apply(out);
		out = tf.layers.reLU().apply(out);

		out = tf.layers.conv2d({
			filters: filters,
			kernelSize: 3,
			strides: 1,
			padding: 'same',
			useBias: false
		}).apply(out);
		
		out = tf.layers.batchNormalization().apply(out);
		
		// Add the shortcut connection
		out = tf.layers.add().apply([out, shortcut]);
		out = tf.layers.reLU().apply(out);

		return out;
	}
	
	// buildModel() {
	// 	const model = tf.sequential()

	// 	// convert from flattened to tensor
	// 	model.add(tf.layers.reshape({
	// 		targetShape: [28, 28, 3], // data shape for bloodMNIST
	// 		inputShape: [2352]
	// 	}))

	// 	// first conv2d layer
	// 	model.add(tf.layers.conv2d({
	// 		filters: 32,
	// 		kernelSize: 3,
	// 		strides: 1,
	// 		padding: 'same',
	// 		activation: 'relu'
	// 	}))

	// 	// residual blocks, 1
	// 	this.addResidualBlock(model, 32)
	// 	this.addResidualBlock(model, 32)

	// 	// second set of blocks plus downsampling layer
	// 	model.add(tf.layers.conv2d({
	// 		filters: 64,
	// 		kernelSize: 3,
	// 		strides: 2,
	// 		padding: 'same'
	// 	}))
	// 	this.addResidualBlock(model, 64)
	// 	this.addResidualBlock(model, 64)

	// 	// third set of blocks with another downsampling layer
	// 	model.add(tf.layers.conv2d({
	// 		filters: 128,
	// 		kernelSize: 3,
	// 		strides: 2,
	// 		padding: 'same'
	// 	}))
	// 	this.addResidualBlock(model, 128)

	// 	// pooling and softmax
	// 	model.add(tf.layers.globalAveragePooling2d({
	// 		dataFormat: 'channelsLast'
	// 	}))
	// 	model.add(tf.layers.dense({
	// 		units: 8,
	// 		activation: 'softmax'
	// 	}))

	// 	// compile model
	// 	model.compile({
	// 		optimizer: 'adam',
	// 		loss: 'categoricalCrossentropy',
	// 		metrics: ['accuracy']
	// 	})

	// 	addLog('model initialized.')
	// 	return model
	// }

	// // helper for the residual blocks
	// addResidualBlock(model, filters) {
	// 	const input = model.layers[model.layers.length - 1].output

	// 	// first convolution
	// 	const conv1 = tf.layers.conv2d({
	// 		filters: filters,
	// 		kernelSize: 3,
	// 		padding: 'same',
	// 		activation: 'relu'
	// 	}).apply(input)

	// 	const conv2 = tf.layers.conv2d({
	// 		filters: filters,
	// 		kernelSize: 3,
	// 		padding: 'same'
	// 	}).apply(conv1)

	// 	tf.layers.add().apply([input, conv2])

	// 	model.add(tf.layers.activation({ activation: 'relu' }))
	// }

	forward(x) {
		return super.forward(x, [1, 2352])
	}

	async train(dataSet, config = {
		epochs: 2,
		batchSize: 16,
		validationSplit: 0.2,
		shuffle: true,
		verbose: 1
	}) {
		// take raw array of values and turn to tensor
		const images = tf.tensor2d(dataSet.images, [dataSet.images.length, 2352])

		const labels = tf.oneHot(tf.tensor1d(dataSet.labels, 'int32'), 8)

		// create config object
		const trainingConfig = {
			epochs: config.epochs,
			batchSize: config.batchSize,
			validationSplit: config.validationSplit,
			shuffle: config.shuffle,
			verbose: config.verbose,
			callbacks: {
				// callback in between epochs
				onEpochEnd: (epoch, logs) => {
					addLog(`Epoch ${epoch + 1}`)
					addLog(`Loss: ${logs.loss.toFixed(4)}`)
					addLog(`Accuracy: ${(logs.acc * 100).toFixed(2)}%`)
					if (logs.val_loss) {
						addLog(`  Validation Loss: ${logs.val_loss.toFixed(4)}`)
						addLog(`  Validation Accuracy: ${(logs.val_acc * 100).toFixed(2)}%`)
					}
				}
			}
		}

		try {
			addLog(`Beginning training...`)
			const history = await this.model.fit(images, labels, trainingConfig)
			addLog(`Training completed`)

			images.dispose()
			labels.dispose()

			return history
		} catch (error) {
			console.error('Error during training: ', error)

			images.dispose()
			labels.dispose()
			throw error
		}
	}
}
