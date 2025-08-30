// colab link:  [...]

function MnistRNN() {
    var model = this;

    this.weights_meta = {
        '(MnistNet).dropout(Dropout).keygen(Generator)._key': [[1973249, 1973251], [2]],
        '(MnistNet).lstm_core(LSTMCore).fc(Linear).b': [[266496, 268544], [2048]],
        '(MnistNet).lstm_core(LSTMCore).fc(Linear).w': [[268544, 1841408], [768, 2048]],
        '(MnistNet).output_head(Linear).b': [[1841408, 1841665], [257]],
        '(MnistNet).output_head(Linear).w': [[1841665, 1973249], [512, 257]],
        '(MnistNet).pos_embed(Embed).embeddings': [[0, 200704], [784, 256]],
        '(MnistNet).value_embed(Embed).embeddings': [[200704, 266496], [257, 256]]
    };

    this.is_model_ready = false;
    
    this.embed_lookup = function(index, weights) {
        return tf.slice(weights, [index], [1]);
    };

    this.pos = 0;
    this.state = null;
    this.start_token = 256;
    this.hidden_size = this.weights_meta['(MnistNet).lstm_core(LSTMCore).fc(Linear).b'][1][0] / 4;

    this.initialize_state = function() {
        this.pos = 0;
        this.token = this.start_token;
        var hidden = tf.zeros([1, this.hidden_size]);
        var cell = tf.zeros([1, this.hidden_size]);
        this.state = [hidden, cell];
    };

    this.lstm_core = function(inputs, state, weights) {
        const [hidden, cell] = state;
        const [w, b] = weights;
        const i_and_h =tf.concat([inputs, hidden], 1);
        const gated = tf.add(tf.matMul(i_and_h, w), b);
        const [i, g, f, o] = tf.split(gated, 4, 1);
        const f_ = tf.sigmoid(tf.add(f, 1.));
        const i_ = tf.sigmoid(i);
        const g_ = tf.tanh(g);
        const c = tf.add(
            tf.mul(i_, g_),
            tf.mul(cell, f_)
        );
        const h = tf.mul(
            tf.sigmoid(o),
            tf.tanh(c)
        );
        return [h, c];
    };

    this.step = function() {
        const [token, h, c] = tf.tidy( function() {
            const lstm_b = model.MODEL_WEIGHTS['(MnistNet).lstm_core(LSTMCore).fc(Linear).b'];
            const lstm_w = model.MODEL_WEIGHTS['(MnistNet).lstm_core(LSTMCore).fc(Linear).w'];
            const output_b = model.MODEL_WEIGHTS['(MnistNet).output_head(Linear).b'];
            const output_w = model.MODEL_WEIGHTS['(MnistNet).output_head(Linear).w'];
            const pos_embed = model.MODEL_WEIGHTS['(MnistNet).pos_embed(Embed).embeddings'];
            const value_embed = model.MODEL_WEIGHTS['(MnistNet).value_embed(Embed).embeddings'];
            const v = model.embed_lookup(model.token, value_embed);
            const p = model.embed_lookup(model.pos, pos_embed);
            const x = tf.add(v, p);
            const [h, c] = model.lstm_core(x, model.state, [lstm_w, lstm_b]);
            tf.dispose(model.state[0]);
            tf.dispose(model.state[1]);
            const logits = tf.add(
                tf.matMul(h, output_w),
                output_b
            );
            const token = tf.multinomial(logits, 1).dataSync()[0];
            
            return [token, h, c];
        });

        this.clean_memory();
        this.token = token;
        this.state = [h, c];
        canvas.plot_xyc(this.pos, token);
        this.pos = this.pos + 1;
    };

    this.MODEL_WEIGHTS = {};
    this.clean_memory = function() {
        tf.dispose(model.state[0]);
        tf.dispose(model.state[1]);
    };

    this.loop = function() {
        this.step();
        if (this.pos >=28*28) {
            setTimeout(function(){
                model.clean_memory();
                model.initialize_state();
                canvas.plot_grid();
                model.loop();
            }, 3000);
        } else {
            canvas.plot_xyc(this.pos, 255);
            setTimeout(function(){model.loop();}, 0);
        }
    };
    
    this.load_model_weights = function() {
        var req = new XMLHttpRequest();
        req.open("GET", "weights.bin", true);
        console.log('loading weights...');
        req.responseType = "arraybuffer";
        var this_ = this;
        req.onload = function (event) {
            var buff = req.response;
            if (buff) {
                var W = new Float32Array(buff);
                for(var k in this_.weights_meta) {
                    info = this_.weights_meta[k];
                    offset = info[0];
                    shape = info[1];
                    this_.MODEL_WEIGHTS[k] = tf.tensor(W.subarray(offset[0], offset[1]), shape);
                }
                this_.is_model_ready = true;
            } else {
                alert('Error while loading weights...');
            }
        };
        req.send(null);
    };

    this.load_when_ready = function() {
        tf.ready().then( function() {
            tf.enableProdMode();
            console.log('tf is ready');
            model.initialize_state()
            model.load_model_weights();
            console.log(model.hidden_size);
        });    
    };
}


function MnistCanvas() {
    var canvas = document.getElementById("mnist-canvas");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    context=canvas.getContext('2d');
    context.translate(canvas.width/2,canvas.height/2);
    var scale = Math.floor(Math.min(canvas.width, canvas.height) / (28*2) ) * 28;
    console.log(scale);
    context.scale(scale, scale)
    context.imageSmoothingEnabled = false;

    this.clear = function() {
        context.clearRect(-1, -1, 2., 2.);
        context.fillStyle = "rgb(0, 0, 0)";
        context.fillRect(-10, -10, 20, 20);
    };

    this.plot_grid = function() {
        for (var i=0; i< 28*28; i++) this.plot_xyc(i, 0);
    };

    this.plot_xyc = function (pos, color) {
        color = Math.max(20, color);
        var step = 1. / 28;
        var y = Math.floor(pos / 28 - 14) * step;
        var x = (pos % 28 - 14) * step;
        context.fillStyle = "rgb(0, " + color + ", 0)";
        context.fillRect(x, y, step, step);
        context.strokeStyle = "rgb(0, 0, 0)";
        context.lineWidth = 0.008;
        context.strokeRect(x, y, step, step);
    };

    this.loading_animation = function() {
        var counter = 0;
        var this_ = this;
        this_.plot_grid();

        var draw = function()  {
            if (model.is_model_ready) {
                console.log('stopping animation.');
                model.loop();
                return;
            }
            if (counter >= 28*28) {
                this_.plot_grid();
                counter = 0;
            }
            this_.plot_xyc(counter, 255);
            if (counter < 28*28-1) {
                this_.plot_xyc(counter+1, 255);
            }
            counter = counter+1;
            window.requestAnimationFrame(draw);
        };
        window.requestAnimationFrame(draw);
    };    
}


var model = null; 
var canvas = null;

window.onload = function() {
    setTimeout(function() {
        model = new MnistRNN();
        canvas = new MnistCanvas();
        console.log("init...");
        canvas.clear();
        canvas.loading_animation();
        model.load_when_ready();
    }, 500);
};
