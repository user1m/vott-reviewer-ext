import os

from reviewers import app
# from home import app

if __name__ == '__main__':
    app.debug = True
    host = os.environ.get('IP', '0.0.0.0')
    port = int(os.environ.get('PORT', 80))
    # app.run(use_reloader=True)
    app.run(host=host, port=port)
