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

	startTrainingBtn.disabled = false
	loadDataBtn.disabled = false

	addLog("Done.")
})