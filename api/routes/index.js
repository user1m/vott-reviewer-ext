'use strict';

var home = require('../controllers/home');
var cntk = require('../controllers/cntk');

module.exports = function (app) {
  app.route('/')
    .get(home);

  app.route('/cntk')
    .get(cntk.get)

  app.route('/cntk/review')
    .post(cntk.post);
};
