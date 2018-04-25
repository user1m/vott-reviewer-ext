'use strict';

const express = require('express'),
    app = express(),
    port = process.env.PORT || 80,
    bodyParser = require('body-parser'),
    cors = require('cors');

app.options('*', cors());
app.use(cors());

app.use(bodyParser.urlencoded({
    extended: true
}));

app.use(bodyParser.json());

const routes = require('./routes/index');
routes(app);

app.listen(port);

console.log('API server started on: ' + port);
