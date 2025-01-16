import * as tf from '@tensorflow/tfjs'

class Model {
	constructor () {
	}
	
	summary () {
		this.model.summary()
	}

	forward(x, shape) {
		const tensor = (x instanceof tf.Tensor) ? x : tf.tensor2d([x], shape)
		const output = this.model.predict(tensor)
		console.log('Output:', output.arraySync())
		return output
	}
}

// test model
export class testModel extends Model {
	constructor () {
		super()
		console.log("Initializing test model instance...")
		this.model = this.buildModel()
	}

	buildModel () {
		// make a sequential model
		this.model = tf.sequential();

		// add the feed forward test layer
		this.model.add(tf.layers.dense({
			units: 2,
			inputShape: [2],
			activation: 'linear'
		}))

		this.model.compile({
			optimizer: 'adam',
			loss: 'meanSquaredError'
		})

		return this.model
	}
}

// resnet
export class ResNet10 extends Model {
	constructor () {
		super()
		console.log("Initializing ResNet10 instance...")
		this.model = this.buildModel()
	}

	buildModel () {
		const model = tf.sequential()

		// convert from flattened to tensor
		model.add(tf.layers.reshape({
			targetShape: [28, 28, 3], // data shape for bloodMNIST
			inputShape: [2352]
		}))

		// first conv2d layer
		model.add(tf.layers.conv2d({
			filters: 32,
			kernelSize: 3,
			strides: 1,
			padding: 'same',
			activation: 'relu'
		}))

		// residual blocks, 1
		this.addResidualBlock(model, 32)
		this.addResidualBlock(model, 32)

		// second set of blocks plus downsampling layer
		model.add(tf.layers.conv2d({
			filters: 64,
			kernelSize: 3,
			strides: 2,
			padding: 'same'
		}))
		this.addResidualBlock(model, 64)
		this.addResidualBlock(model, 64)

		// third set of blocks with another downsampling layer
		model.add(tf.layers.conv2d({
			filters: 128,
			kernelSize: 3,
			strides: 2,
			padding: 'same'
		}))
		this.addResidualBlock(model, 128)

		// pooling and softmax
		model.add(tf.layers.globalAveragePooling2d({
			dataFormat: 'channelsLast'
		}))
		model.add(tf.layers.dense({
			units: 8,
			activation: 'softmax'
		}))

		// compile model
		model.compile({
			optimizer: 'adam', 
			loss: 'categoricalCrossentropy',
			metrics: ['accuracy']
		})

		return model
	}

	// helper for the residual blocks
	addResidualBlock(model, filters) {
		const input = model.layers[model.layers.length - 1].output

		// first convolution
		const conv1 = tf.layers.conv2d({
			filters: filters,
			kernelSize: 3,
			padding: 'same',
			activation: 'relu'
		}).apply(input)

		const conv2 = tf.layers.conv2d({
			filters: filters,
			kernelSize: 3,
			padding: 'same'
		}).apply(conv1)

		tf.layers.add().apply([input, conv2])

		model.add(tf.layers.activation({activation: 'relu'}))
	}

	forward(x) {
		return super.forward(x, [1, 2352])
	}
}
