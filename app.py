import os
import json
from flask import Flask, jsonify, request, make_response, render_template

BASE_DIR = os.path.dirname(os.getcwd())
templates_dir = os.path.join(BASE_DIR, 'templates')
static_dir = os.path.join(BASE_DIR, 'static')
app = Flask(__name__)

# 导入分词工具
from cut_word.predict import predict, switch_cut_words
print('cut tool success')


@app.route('/', methods=['GET', 'POST'])
def index():
    print('##### response index----------')
    return render_template('index.html')


@app.route('/cut_words/', methods=['GET', 'POST'])
def cut_words():
    print('##### response cut_words----------')
    ret = {}
    if request.method == 'GET':
        str_list = request.args.get('str_list')
        if type(str_list) == str:
            str_list = [str_list]
            tags = predict(str_list)
            cut_words = switch_cut_words(str_list, tags)
            ret['result'] = 0
            ret['msg'] = 'success'
            ret['data'] = cut_words
        return json.dumps(ret, ensure_ascii=False)


if __name__ == '__main__':
    app.run('127.0.0.1', port=80)







