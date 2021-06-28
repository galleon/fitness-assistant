import React, { useEffect, useState, useRef } from "react";
import { Grid, AppBar, Toolbar, Typography, Button, Card, CardContent, CardActions } from '@material-ui/core';
import { FormControl, InputLabel, NativeSelect, FormHelperText, Snackbar, CircularProgress } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import MuiAlert from '@material-ui/lab/Alert';
import './App.css';

import * as tf from '@tensorflow/tfjs';
import * as posenet from "@tensorflow-models/posenet";
import '@tensorflow/tfjs-backend-webgl';
import Webcam from "react-webcam";
import { drawKeypoints, drawSkeleton } from "./utilities";

import { processData } from "./dataProcessing";
import { runTraining } from './modelTraining';

function Alert(props) {
    return <MuiAlert elevation={6} variant="filled" {...props} />;
}

const useStyles = makeStyles((theme) => ({
    backgroundAppBar: {
        background: '#1875d2'
    },
    title: {
        flexGrow: 1,
        textAlign: 'left'
    },
    statsCard: {
        width: '250px',
        margin: '10px',
    },
    singleLine: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
    },
    formControl: {
        margin: theme.spacing(1),
        minWidth: 120,
    }
}));

const delay = (time) => {
    return new Promise((resolve, reject) => {
        if (isNaN(time)) {
            reject(new Error('delay requires a valid number.'));
        } else {
            setTimeout(resolve, time);
        }
    });
}

function App() {
    const classes = useStyles();

    const webcamRef = useRef(null);
    const canvasRef = useRef(null);
    const [model, setModel] = useState(null);
    const poseEstimationLoop = useRef(null);
    const [isPoseEstimation, setIsPoseEstimation] = useState(false)
    const [opCollectData, setOpCollectData] = useState('inactive');
    const [snackbarDataColl, setSnackbarDataColl] = useState(false);
    const [snackbarDataNotColl, setSnackbarDataNotColl] = useState(false);
    const [snackbarTrainingError, setSnackbarTrainingError] = useState(false);

    const [dataCollect, setDataCollect] = useState(false);
    const [trainModel, setTrainModel] = useState(false);
    const [rawData, setRawData] = useState([]);

    const windowWidth = 400;
    const windowHeight = 300;

    let state = 'waiting';

    const openSnackbarDataColl = () => {
        setSnackbarDataColl(true);
    };

    const closeSnackbarDataColl = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        setSnackbarDataColl(false);
    };

    const openSnackbarDataNotColl = () => {
        setSnackbarDataNotColl(true);
    };

    const closeSnackbarDataNotColl = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        setSnackbarDataNotColl(false);
    };

    const openSnackbarTrainingError = () => {
        setSnackbarTrainingError(true);
    };

    const closeSnackbarTrainingError = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        setSnackbarTrainingError(false);
    };

    const [workoutState, setWorkoutState] = useState({
        workout: '',
        name: 'hai',
    });

    useEffect(() => {
        loadPosenet();
    }, []);

    const collectData = async () => {
        setOpCollectData('active');
        await delay(10000);

        openSnackbarDataColl();
        console.log('collecting');
        state = 'collecting';

        await delay(30000);

        openSnackbarDataNotColl();
        console.log('not collecting');
        state = 'waiting';

        setOpCollectData('inactive');
    };

    const loadPosenet = async () => {
        let loadedModel = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            inputResolution: { width: 400, height: 300 },
            multiplier: 0.75
        });

        setModel(loadedModel)
        console.log("Posenet Model Loaded..")
    };

    const startPoseEstimation = () => {
        if (
            typeof webcamRef.current !== "undefined" &&
            webcamRef.current !== null &&
            webcamRef.current.video.readyState === 4
        ) {
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
                model.estimateSinglePose(video, {
                    flipHorizontal: false
                }).then(pose => {
                    var toc = new Date().getTime();

                    let inputs = [];

                    for (let i = 0; i < pose.keypoints.length; i++) {
                        let x = pose.keypoints[i].position.x;
                        let y = pose.keypoints[i].position.y;
                        if (pose.keypoints[i].score < 0.1) {
                            x = 0;
                            y = 0;
                        } else {
                            x = (x / (windowWidth / 2)) - 1;
                            y = (y / (windowHeight / 2)) - 1;
                        }
                        inputs.push(x);
                        inputs.push(y);
                    }

                    console.log('STATE->' + state);
                    if (state === 'collecting') {
                        console.log(toc - tic, " ms");
                        console.log(tf.getBackend());
                        console.log(pose);
                        console.log(workoutState.workout);

                        const rawDataRow = { xs: inputs, ys: workoutState.workout };
                        rawData.push(rawDataRow);
                        setRawData(rawData);
                    }

                    drawCanvas(pose, videoWidth, videoHeight, canvasRef);
                });
            }, 100);
        }
    };

    const drawCanvas = (pose, videoWidth, videoHeight, canvas) => {
        const ctx = canvas.current.getContext("2d");
        canvas.current.width = videoWidth;
        canvas.current.height = videoHeight;

        drawKeypoints(pose["keypoints"], 0.5, ctx);
        drawSkeleton(pose["keypoints"], 0.5, ctx);
    };

    const stopPoseEstimation = () => clearInterval(poseEstimationLoop.current);

    const handlePoseEstimation = (input) => {
        if (input === 'COLLECT_DATA') {
            if (isPoseEstimation) {
                if (opCollectData === 'inactive') {
                    setIsPoseEstimation(current => !current);
                    stopPoseEstimation();
                    state = 'waiting';
                    setDataCollect(false);
                }
            } else {
                if (workoutState.workout.length > 0) {
                    setIsPoseEstimation(current => !current);
                    startPoseEstimation();
                    collectData();
                    setDataCollect(true);
                }
            }
        }
    };

    const handleWorkoutSelect = (event) => {
        const name = event.target.name;
        setWorkoutState({
            ...workoutState,
            [name]: event.target.value,
        });
    };

    const handleTrainModel = async () => {
        if (rawData.length > 0) {
            console.log('Data size: ' + rawData.length);
            setTrainModel(true);
            const [numOfFeatures, convertedDatasetTraining, convertedDatasetValidation] = processData(rawData);
            await runTraining(convertedDatasetTraining, convertedDatasetValidation, numOfFeatures);
            setTrainModel(false);
        } else {
            openSnackbarTrainingError();
        }
    }

    return (
        <div className="App">
            <Grid container spacing={3}>
                <Grid item xs={12}>
                    <AppBar position="static" className={classes.backgroundAppBar}>
                        <Toolbar variant="dense">
                            <Typography variant="h6" color="inherit" className={classes.title}>
                                Doctor Assistant
                            </Typography>
                            <Button color="inherit">Start Workout</Button>
                            <Button color="inherit">History</Button>
                            <Button color="inherit">Reset</Button>
                        </Toolbar>
                    </AppBar>
                </Grid>
            </Grid>
            <Grid container spacing={3}>
                <Grid item xs={12}>
                    <Card>
                        <CardContent>
                            <Webcam
                                ref={webcamRef}
                                style={{
                                    marginTop: "10px",
                                    marginBottom: "10px",
                                    marginLeft: "auto",
                                    marginRight: "auto",
                                    left: 0,
                                    right: 0,
                                    textAlign: "center",
                                    zindex: 9,
                                    width: 400,
                                    height: 300,
                                }}
                            />
                            <canvas
                                ref={canvasRef}
                                style={{
                                    marginTop: "10px",
                                    marginBottom: "10px",
                                    position: "absolute",
                                    marginLeft: "auto",
                                    marginRight: "auto",
                                    left: 0,
                                    right: 0,
                                    textAlign: "center",
                                    zindex: 9,
                                    width: 400,
                                    height: 300,
                                }}
                            />
                        </CardContent>
                        <CardActions style={{ justifyContent: 'center' }}>
                            <Grid container spacing={0}>
                                <Grid item xs={12}>
                                    <Toolbar style={{ justifyContent: 'center' }}>
                                        <Card className={classes.statsCard}>
                                            <CardContent>
                                                <Typography className={classes.title} color="textSecondary" gutterBottom>
                                                    Jumping Jacks
                                                </Typography>
                                                <Typography variant="h2" component="h2" color="secondary">
                                                    75
                                                </Typography>
                                            </CardContent>
                                        </Card>
                                        <Card className={classes.statsCard}>
                                            <CardContent>
                                                <Typography className={classes.title} color="textSecondary" gutterBottom>
                                                    Wall-Sit
                                                </Typography>
                                                <Typography variant="h2" component="h2" color="secondary">
                                                    200
                                                </Typography>
                                            </CardContent>
                                        </Card>
                                        <Card className={classes.statsCard}>
                                            <CardContent>
                                                <Typography className={classes.title} color="textSecondary" gutterBottom>
                                                    Lunges
                                                </Typography>
                                                <Typography variant="h2" component="h2" color="secondary">
                                                    5
                                                </Typography>
                                            </CardContent>
                                        </Card>
                                    </Toolbar>
                                </Grid>
                                <Grid item xs={12} className={classes.singleLine}>
                                    <FormControl className={classes.formControl} required>
                                        <InputLabel htmlFor="age-native-helper">Workout</InputLabel>
                                        <NativeSelect
                                            value={workoutState.workout}
                                            onChange={handleWorkoutSelect}
                                            inputProps={{
                                                name: 'workout',
                                                id: 'age-native-helper',
                                            }}>
                                            <option aria-label="None" value="" />
                                            <option value={'JUMPING_JACKS'}>Jumping Jacks</option>
                                            <option value={'WALL_SIT'}>Wall-Sit</option>
                                            <option value={'LUNGES'}>Lunges</option>
                                        </NativeSelect>
                                        <FormHelperText>Select training data type</FormHelperText>
                                    </FormControl>
                                    <Toolbar>
                                        <Typography style={{ marginRight: 16 }}>
                                            <Button variant="contained" onClick={() => handlePoseEstimation('COLLECT_DATA')} color={isPoseEstimation ? 'secondary' : 'default'}
                                                disabled={trainModel}>
                                                {isPoseEstimation ? "Stop" : "Collect Data"}
                                            </Button>
                                        </Typography>
                                        <Typography style={{ marginRight: 16 }}>
                                            <Button variant="contained" onClick={() => handleTrainModel()} disabled={dataCollect}>
                                                Train Model
                                            </Button>
                                        </Typography>
                                        {trainModel ? <CircularProgress color="secondary" /> : null}
                                    </Toolbar>
                                </Grid>
                            </Grid>
                        </CardActions>
                    </Card>
                </Grid>
            </Grid>
            <Snackbar open={snackbarDataColl} autoHideDuration={2000} onClose={closeSnackbarDataColl}>
                <Alert onClose={closeSnackbarDataColl} severity="info">
                    Started collecting pose data!
                </Alert>
            </Snackbar>
            <Snackbar open={snackbarDataNotColl} autoHideDuration={2000} onClose={closeSnackbarDataNotColl}>
                <Alert onClose={closeSnackbarDataNotColl} severity="success">
                    Completed collecting pose data!
                </Alert>
            </Snackbar>
            <Snackbar open={snackbarTrainingError} autoHideDuration={2000} onClose={closeSnackbarTrainingError}>
                <Alert onClose={closeSnackbarTrainingError} severity="error">
                    Training data is not available!
                </Alert>
            </Snackbar>
        </div>
    );
}

export default App;
