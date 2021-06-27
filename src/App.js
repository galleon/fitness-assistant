import React, { useEffect, useState, useRef } from 'react';
import * as posenet from '@tensorflow-models/posenet'
// import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import Webcam from 'react-webcam';

// import logo from './logo.svg';
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


function App() {
    const [posenetModel, setModel] = useState({});
    const webcamRef = useRef({});
    const poseEstimationLoop = useRef(null);
    const [isPoseEstimation, setIsPoseEstimation] = useState(false)

    useEffect(() => {
        loadPosenetModel();
    }, []);

    const videoConstraints = {
        width: 800,
        height: 600,
        position: "absolute",
        marginLeft: "auto",
        marginRight: "auto",
        left: 0,
        right: 0,
        textAlign: "center",
        zindex: 9
    };

    const loadPosenetModel = async () => {
        // Load the PoseNet model
        let posenetModel = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            inputResolution: { width: 800, height: 600 },
            multiplier: 0.75,
            quantBytes: 4
        });

        setModel(posenetModel);
        console.log("Posenet model loaded...");
        console.log(posenetModel.summary);
    };

    const startPoseEstimation = () => {
        if (webcamRef && webcamRef.current.video.readyState === 4) {
            // Run pose estimation each 100 milliseconds
            poseEstimationLoop.current = setInterval(() => {
                // Get Video Properties
                const video = webcamRef.current.video;
                const videoWidth = webcamRef.current.video.videoWidth;
                const videoHeight = webcamRef.current.video.videoHeight;

                // Set video width
                webcamRef.current.video.width = videoWidth;
                webcamRef.current.video.height = videoHeight;

                // Do pose estimation
                var tic = new Date().getTime()
                posenetModel.estimateSinglePose(video, {
                    flipHorizontal: false
                }).then(pose => {
                    // What to do here
                    console.log(tic)
                    console.log(pose.summary)
                });
            }, 100);
        }
    };

    const stopPoseEstimation = () => clearInterval(poseEstimationLoop.current);

    const handlePoseEstimation = () => {
        // Is poseEstimation is running
        if (isPoseEstimation === true) {
            stopPoseEstimation();
            setIsPoseEstimation(false);
        } else {
            startPoseEstimation();
            setIsPoseEstimation(true);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                />
                <button
                    style={{
                        position: "relative",
                        marginLeft: "auto",
                        marginRight: "auto",
                        top: 320,
                        left: 0,
                        right: 0,
                        textAlign: "center",
                        zindex: 9
                    }}
                    onClick={handlePoseEstimation}>
                    {isPoseEstimation ? "Stop" : "Start"}
                </button>

            </header>
        </div>
    );
}

export default App;
