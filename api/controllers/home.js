'use strict';

module.exports = function home(req, res, next) {
    let msg = `<h1>Welcome To VOTT Reviewer Service</h1>`;
    msg += `<br/> <p>Apis available:</p>`;
    msg += `<ul><li>get: /</li><li>post: /{mlframework}/review</li></ul>`;
    res.send(msg);
};

