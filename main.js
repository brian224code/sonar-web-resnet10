import { addLog, getTrainingConfig } from './util.js'
import { ResNet10 } from './models.js'


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
	const weightsJSON = {
		weights: [],
		shapes: [],
		names: []
	};
	const layers = model.model.layers
	addLog(`Model layers: ${layers.length}`)

	for (let i = 0; i < layers.length; i++) {
		const layer = layers[i]
		const layerWeights = layer.getWeights()

		for (let j = 0; j < layerWeights.length; j++) {
			const weightTensor = layerWeights[j]
			const weightValues = await weightTensor.data(0)

			weightsJSON.weights.push(Array.from(weightValues))
			weightsJSON.shapes.push(weightTensor.shape)
			weightsJSON.names.push(`${layer.name}_weight_${j}`)
		}
	}

	addLog(`Done processing. Downloading...`)

	const weightString = JSON.stringify(weightsJSON, null, 2)

	const blob = new Blob([weightString], { type: 'application/json' })
	const url = URL.createObjectURL(blob)
	const a = document.createElement('a')
	a.href = url
	a.downoad = 'model_weights.json'
	document.body.appendChild(a)
	a.click()
	document.body.removeChild(a)
	URL.revokeObjectURL(url)
})