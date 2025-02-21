import { addLog, getTrainingConfig } from './util.js'
import { ResNet10 } from './models.js'
import * as tf from '@tensorflow/tfjs'


// dataset loading helper
async function loadDataset(jsonFile) {
	try {
		const response = await fetch(jsonFile)
		const jsonData = await response.json()
		return processData(jsonData)
	} catch (error) {
		console.error('Error loading data:', error)
	}
}

// data processing helper
function processData(jsonData) {
	const images = jsonData.map(item => item.image)
	const labels = jsonData.map(item => item.label)

	return {
		images: images,
		labels: labels
	}
}


// globals
let model = null
let dataSet = null

const loadDataBtn = document.getElementById('loadData')
const startTrainingBtn = document.getElementById('startTraining')
const dumpWeightsBtn = document.getElementById('dumpWeights')

// initial page loading
document.addEventListener('DOMContentLoaded', () => {
	addLog('Page loaded.')
})

// handle dataloading
loadDataBtn.addEventListener('click', async () => {
	addLog('Loading data...')
	dataSet = await loadDataset('/data/bloodmnist_test.json')
	addLog('Data loaded.')

	startTrainingBtn.disabled = false
})

startTrainingBtn.addEventListener('click', async () => {
	// create model
	model = new ResNet10()
	// process config from html
	let trainingConfig = getTrainingConfig()

	try {
		startTrainingBtn.disabled = true
		loadDataBtn.disabled = true

		await model.train(dataSet, trainingConfig)
	} catch (error) {
		addLog(`Error during training:`, error)
	}

	dumpWeightsBtn.disabled = false
	
	addLog("Done.")
})

dumpWeightsBtn.addEventListener('click', async () => {
	addLog("Dumping weights...")
	// const weightsJSON = {
	// 	weights: [],
	// 	shapes: [],
	// 	names: []
	// };
	// const layers = model.model.layers
	// addLog(`Model layers: ${layers.length}`)

	// for (let i = 0; i < layers.length; i++) {
	// 	const layer = layers[i]
	// 	const layerWeights = layer.getWeights()

	// 	for (let j = 0; j < layerWeights.length; j++) {
	// 		const weightTensor = layerWeights[j]
	// 		const weightValues = await weightTensor.data(0)

	// 		weightsJSON.weights.push(Array.from(weightValues))
	// 		weightsJSON.shapes.push(weightTensor.shape)
	// 		weightsJSON.names.push(`${layer.name}_weight_${j}`)
	// 	}
	// }

	// const weightString = JSON.stringify(weightsJSON, null, 2)

	const weights = model.model.getWeights();
	const weightString = await serializeMessage(weights);

	addLog(`Done processing. Downloading...`)

	const blob = new Blob([weightString], { type: 'application/json' })
	const url = URL.createObjectURL(blob)
	const a = document.createElement('a')
	a.href = url
	a.download = 'model_weights.json'
	document.body.appendChild(a)
	a.click()
	document.body.removeChild(a)
	URL.revokeObjectURL(url)
})

async function tensorToSerializable(obj) {
    if (obj instanceof tf.Tensor) {
        return {
            "__tensor__": true,
            "data": await obj.array(),
            "dtype": obj.dtype,
            "shape": obj.shape
        };
    } else if (Array.isArray(obj)) {
        return Promise.all(obj.map(item => tensorToSerializable(item)));
    } else if (typeof obj === "object" && obj !== null) {
        const entries = await Promise.all(
            Object.entries(obj).map(async ([key, value]) => [key, await tensorToSerializable(value)])
        );
        return Object.fromEntries(entries);
    }
    return obj;
}

function serializableToTensor(obj) {
    if (typeof obj === "object" && obj !== null) {
        if ("__tensor__" in obj) {
            return tf.tensor(obj.data, obj.shape, obj.dtype);
        }
        return Object.fromEntries(Object.entries(obj).map(([key, value]) => [key, serializableToTensor(value)]));
    } else if (Array.isArray(obj)) {
        return obj.map(item => serializableToTensor(item));
    }
    return obj;
}

async function serializeMessage(message) {
    const serializableDict = await tensorToSerializable(message);
    return JSON.stringify(serializableDict, null, 2);
}

function deserializeMessage(jsonStr) {
	const serializableDict = JSON.parse(json);
	return serializableToTensor(serializableDict);
}
