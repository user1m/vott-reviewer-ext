'use strict';

const uuidv1 = require("uuid/v1"); //timestamp
const id = uuidv1();
const inputFilePath = `/workdir/output/inputs/${id}.jpg`;
const input = "/cntk/Examples/Image/DataSets/Grocery/grocery/positive/WIN_20160803_11_29_07_Pro.jpg";
const modelPath = "/workdir/output/faster_rcnn_eval_AlexNet_e2e.model";
const jsonFilePath = `/workdir/temp/outputs/json-result-${id}.json`;
var fs = require('fs');
// var Promise = require('bluebird');
// var exec = require('child_process').execFile;

function promiseFromChildProcess(child) {
    return new Promise(function (resolve, reject) {
        child.addListener("error", reject);
        child.addListener("exit", resolve);
    });
}

function rmFile(file) {
    return new Promise((resolve, reject) => {
        fs.unlink(file, function (err) {
            if (err && err.code == 'ENOENT') {
                console.info("File doesn't exist, won't remove it.");  // file doens't exist
            } else if (err) {
                console.error("Error occurred while trying to remove file"); // other errors, e.g. maybe we don't have enough permission
            } else {
                console.info(`removed`);
            }
        });
    });
}

function saveImageToDisk(data) {
    return new Promise((resolve, reject) => {
        fs.writeFile(inputFilePath, data, "binary", function (err) {
            if (err) {
                console.log("ERROR!!! SAVING IMAGE TO DISK.....");
                console.log(err);
                reject(err);
            } else {
                console.log("FINISH SAVING IMAGE TO DISK.....");
                console.log(`${inputFilePath}: IMAGE SAVED`);
                resolve(true);
            }
        });
    })
}

function readImageData(req, res) {
    return new Promise((resolve, reject) => {
        let contentType = req.headers["content-type"] || "";
        let mime = contentType.split(";")[0];
        if (req.method == "POST" && mime == "application/octet-stream") {
            console.log("PROCESSING IMAGE RAW DATA.....");
            let data = "";
            req.setEncoding("binary");

            req.on("data", function (chunk) {
                data += chunk;
                // 1e6 === 1 * Math.pow(10, 6) === 1 * 1000000 ~~~ 1MB
                if (data.length > 10 * Math.pow(10, 6)) {
                    console.log("TOO MUCH DATA.....KILLING CONNECTION");
                    res.send(`Image too large! Please upload an image under 10MB`);
                    req.connection.destroy();
                    reject();
                }
            });


            req.on("end", function () {
                console.log("FINISH PROCESSING IMAGE RAW DATA.....");
                resolve(data);
                // next();
            });
        }
    });
}

function execCmd() {
    return new Promise((resolve, reject) => {
        const cmd = `bash -c '/workdir/api/scripts/fast-rcnn.sh ${inputFilePath} ${modelPath} ${jsonFilePath}'`;
        const rmcmd = `rm -rf ${jsonFilePath}'`;
        // var child = exec(cmd);
        // console.log(cmd);
        require('child_process').execSync(cmd, { stdio: [0, 1, 2] });
        const result = fs.readFileSync(jsonFilePath);
        rmFile(jsonFilePath);
        // console.log(result);
        res.send(result);
    });
}

function get(req, res, next) {
    let msg = `<h1> Welcome To VOTT Reviewer Service: <br/> CNTK Endpint</h1> `;
    res.send(msg);
};

function post(req, res, next) {
    //https://stackoverflow.com/questions/30763496/how-to-promisify-nodes-child-process-exec-and-child-process-execfile-functions
    try {
        readImageData(req, res)
            .then((result) => {
                saveImageToDisk(result)
                    .then((result) => {
                        execCmd()
                    })
            });


        // promiseFromChildProcess(child)
        //     .then((result) => {
        //         console.log('promise complete: ' + result);
        //         res.send(fs.readFileSync(outputFile));
        //         exec(`rm ${outputFile} `);
        //     })
        //     .catch((err) => {
        //         console.log('promise rejected: ' + err);
        //         res.status(400).send("ERROR! Request Failed!");
        //     });
    } catch (error) {
        // console.log(error);
        res.status(400).send("ERROR! Request Failed!");
    }
};

module.exports = {
    get,
    post
};
