import { useEffect, useState } from 'react';
import * as posenet from '@tensorflow-models/posenet'
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

import logo from './logo.svg';
import './App.css';

/*
async function doTraining(model, xs, ys) {
    const history =
        await model.fit(xs, ys,
            {
                epochs: 200,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        console.log("Epoch:"
                            + epoch
                            + " Loss:"
                            + logs.loss);

                    }
                }
            });
    console.log(history.params);
}

const handleRunTraining = (event) => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({ optimizer: tf.train.sgd(0.1), loss: 'meanSquaredError' });
    model.summary();

    // Equation: y = 2x - 1
    const xs = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]);
    const ys = tf.tensor2d([-3.0, -1.0, 2.0, 3.0, 5.0, 7.0], [6, 1]);
    doTraining(model, xs, ys).then(() => {
        const prediction = model.predict(tf.tensor2d([10], [1, 1]));
        var res = prediction.dataSync()[0];
        prediction.dispose();

        console.log('Result: ' + res);
    });
} */

async function loadPosenetModel() {
    // model_path: 'localstorage://my-model'
    const model = useState(0)

    useEffect(() =>
        model = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            inputResolution: { width: 800, height: 600 },
            multiplier: 0.75,
            quantBytes: 4
        })
    );

    return model
}

function App() {
    model = await loadPosenetModel();

    return (
        <div className="App">
            <header className="App-header">
                <img src={logo} className="App-logo" alt="logo" />
                <p>
                    Edit <code>src/App.js</code> and save to reload.
                </p>
                <a
                    className="App-link"
                    href="https://reactjs.org"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    Learn React
                </a>
                <br />
                <button onClick={handleRunTraining}>Run training</button>
            </header>
        </div>
    );
}

export default App;
