import { ResNet10, testModel } from './models.js'
// import { trainModel } from './train.js'


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

// initial page loading
document.addEventListener('DOMContentLoaded', () => {
	const trainButton = document.getElementById('train-btn')
	const output = document.getElementById('output')

	trainButton.addEventListener('click', async () => {
		output.textContent = 'Loading dataset...'

		const dataSet = await loadDataset("/data/bloodmnist_test.json")

		let shape = dataSet.images[0].length
		console.log("Data loaded. Sample shape: ", shape)

		output.textContent = 'Starting training...'

		// create the model
		const model = new ResNet10()
		model.summary()

		model.train(dataSet)


		output.textContent = 'check log, demo done'
	})
})
