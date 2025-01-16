import * as tf from '@tensorflow/tfjs'

class Model {
	constructor () {
	}
	
	summary () {
		this.model.summary()
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

	forward(x) {
		const tensor = tf.tensor2d(x)
		const output = this.model.predict(tensor)
		console.log('Output:', output.arraySync())
		return output
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
		return {
			// replace with js-pytorch code
			layers: ['foo', 'bar'],
			params: {}
		}
	}

	summary() {
		console.log('ResNet10 Model Summary:', this.model)
	}
}
