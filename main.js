import { ResNet10, testModel } from './models.js'
// import { trainModel } from './train.js'


// dataset loading helper

// function loadDataset() {
//
// }

// initial page loading
document.addEventListener('DOMContentLoaded', () => {
	// dataSet = loadDataset();

	const trainButton = document.getElementById('train-btn')
	const output = document.getElementById('output')

	trainButton.addEventListener('click', () => {
		output.textContent = 'Starting training...'

		// create the model
		const model = new testModel()
		model.summary()
	})
})
