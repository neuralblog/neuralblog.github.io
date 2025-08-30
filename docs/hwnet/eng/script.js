var log = console.log;
var ctx = null;
var canvas = null;
var RNN_SIZE = 512;
var cur_run = 0;

var randn = function() {
    // Standard Normal random variable using Box-Muller transform.
    var u = Math.random() * 0.999 + 1e-5;
    var v = Math.random() * 0.999 + 1e-5;
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

var rand_truncated_normal = function(low, high) {
    while (true) {
        r = randn();
        if (r >= low && r <= high)
            break;
        // rejection sampling.
    }
    return r;
}



var char2idx = {'\x00': 0, ' ': 1, '!': 2, '"': 3, '#': 4, "'": 5, '(': 6, ')': 7, ',': 8, '-': 9, '.': 10, '0': 11, '1': 12, '2': 13, '3': 14, '4': 15, '5': 16, '6': 17, '7': 18, '8': 19, '9': 20, ':': 21, ';': 22, '?': 23, 'A': 24, 'B': 25, 'C': 26, 'D': 27, 'E': 28, 'F': 29, 'G': 30, 'H': 31, 'I': 32, 'J': 33, 'K': 34, 'L': 35, 'M': 36, 'N': 37, 'O': 38, 'P': 39, 'R': 40, 'S': 41, 'T': 42, 'U': 43, 'V': 44, 'W': 45, 'Y': 46, 'a': 47, 'b': 48, 'c': 49, 'd': 50, 'e': 51, 'f': 52, 'g': 53, 'h': 54, 'i': 55, 'j': 56, 'k': 57, 'l': 58, 'm': 59, 'n': 60, 'o': 61, 'p': 62, 'q': 63, 'r': 64, 's': 65, 't': 66, 'u': 67, 'v': 68, 'w': 69, 'x': 70, 'y': 71, 'z': 72};

var gru_core = function(input, weights, state, hidden_size) {
    var [w_h,w_i,b] = weights;
    var [w_h_z,w_h_a] = tf.split(w_h, [2 * hidden_size, hidden_size], 1);
    var [b_z,b_a] = tf.split(b, [2 * hidden_size, hidden_size], 0);
    gates_x = tf.matMul(input, w_i);
    [zr_x,a_x] = tf.split(gates_x, [2 * hidden_size, hidden_size], 1);
    zr_h = tf.matMul(state, w_h_z);
    zr = tf.add(tf.add(zr_x, zr_h), b_z);
    // fix this
    [z,r] = tf.split(tf.sigmoid(zr), 2, 1);
    a_h = tf.matMul(tf.mul(r, state), w_h_a);
    a = tf.tanh(tf.add(tf.add(a_x, a_h), b_a));
    next_state = tf.add(tf.mul(tf.sub(1., z), state), tf.mul(z, a));
    return [next_state, next_state];
};


var generate = function() {
    cur_run = cur_run + 1;
    setTimeout(function() {   
        var counter = 2000;
        tf.disposeVariables();
        
        tf.engine().startScope();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        dojob(cur_run);
    }, 200);

    return false;
}

var dojob = function(run_id) {
    var text = document.getElementById("user-input").value;
    if (text.length == 0) {
        text = "The quick brown fox jumps over the lazy dog";
    }

    var cur_x = 50.;
    var cur_y = 300.;


    log(text);
    original_text = text;
    text = '' + text + '          ' + text;

    text = Array.from(text).map(function(e) {
        return char2idx[e]
    })
    var text_embed = WEIGHTS['rnn/~/embed_1__embeddings'];
    indices = tf.tensor1d(text, 'int32');
    text = text_embed.gather(indices);

    filter = WEIGHTS['rnn/~/conv1_d__w'];
    embed = tf.conv1d(text, filter, 1, 'same');
    bias = tf.expandDims(WEIGHTS['rnn/~/conv1_d__b'], 0);
    embed = tf.add(embed, bias);

    var writer_embed = WEIGHTS['rnn/~/embed__embeddings'];
    var e = document.getElementById("writers");
    var wid = parseInt(e.value);
    // log(wid);

    wid = tf.tensor1d([wid], 'int32');
    wid = writer_embed.gather(wid);
    embed = tf.add(wid, embed);

    // initial state
    var gru0_hx = tf.zeros([1, RNN_SIZE]);
    var gru1_hx = tf.zeros([1, RNN_SIZE]);
    // var gru2_hx = tf.zeros([1, RNN_SIZE]);

    var att_location = tf.zeros([1, 1]);
    var att_context = tf.zeros([1, 73]);

    var input = tf.tensor([[0., 0., 1.]]);

    gru0_w_h = WEIGHTS['rnn/~/lstm_attention_core/~/gru__w_h'];
    gru0_w_i = WEIGHTS['rnn/~/lstm_attention_core/~/gru__w_i'];
    gru0_bias = WEIGHTS['rnn/~/lstm_attention_core/~/gru__b'];

    gru1_w_h = WEIGHTS['rnn/~/lstm_attention_core/~/gru_1__w_h'];
    gru1_w_i = WEIGHTS['rnn/~/lstm_attention_core/~/gru_1__w_i'];
    gru1_bias = WEIGHTS['rnn/~/lstm_attention_core/~/gru_1__b'];
    att_w = WEIGHTS['rnn/~/lstm_attention_core/~/linear__w'];
    att_b = WEIGHTS['rnn/~/lstm_attention_core/~/linear__b'];
    gmm_w = WEIGHTS['rnn/~/linear__w'];
    gmm_b = WEIGHTS['rnn/~/linear__b'];

    ruler = tf.tensor([...Array(text.shape[0]).keys()]);
    var bias = parseInt(document.getElementById("bias").value) / 100 * 3;

    cur_x = 50.;
    cur_y = 400.;
    var path = [];
    var dx = 0.;
    var dy = 0;
    var eos = 1.;
    var counter = 0;


    function loop(my_run_id) {
        if (my_run_id < cur_run) {
            tf.disposeVariables();
            tf.engine().endScope();
            return;
        }

        counter++;
        if (counter < 2000) {
            [att_location,att_context,gru0_hx,gru1_hx,input] = tf.tidy(function() {
                // Attention
                const inp_0 = tf.concat([att_context, input], 1);
                gru0_hx_ = gru0_hx;
                [out_0,gru0_hx] = gru_core(inp_0, [gru0_w_h, gru0_w_i, gru0_bias], gru0_hx, RNN_SIZE);
                tf.dispose(gru0_hx_);
                const att_inp = tf.concat([att_context, input, out_0], 1);
                const att_params = tf.add(tf.matMul(att_inp, att_w), att_b);
                [alpha,beta,kappa] = tf.split(tf.softplus(att_params), 3, 1);
                att_location_ = att_location;
                att_location = tf.add(att_location, tf.div(kappa, 25.));
                tf.dispose(att_location_)

                const phi = tf.mul(alpha, tf.exp(tf.div(tf.neg(tf.square(tf.sub(att_location, ruler))), beta)));
                att_context_ = att_context;
                att_context = tf.sum(tf.mul(tf.expandDims(phi, 2), tf.expandDims(embed, 0)), 1)
                tf.dispose(att_context_);

                const inp_1 = tf.concat([input, out_0, att_context], 1);
                tf.dispose(input);
                gru1_hx_ = gru1_hx;
                [out_1,gru1_hx] = gru_core(inp_1, [gru1_w_h, gru1_w_i, gru1_bias], gru1_hx, RNN_SIZE);
                tf.dispose(gru1_hx_);

                // GMM
                const gmm_params = tf.add(tf.matMul(out_1, gmm_w), gmm_b);
                [x,y,logstdx,logstdy,angle,log_weight,eos_logit] = tf.split(gmm_params, [5, 5, 5, 5, 5, 5, 1], 1);
                // log_weight = tf.softmax(log_weight, 1);
                // log_weight = tf.log(log_weight);
                // log_weight = tf.mul(log_weight, 1. + bias);
                // const idx = tf.multinomial(log_weight, 1).dataSync()[0];
                // log_weight = tf.softmax(log_weight, 1);
                // log_weight = tf.log(log_weight);
                // log_weight = tf.mul(log_weight, 1. + bias);
                const idx = tf.argMax(log_weight, 1).dataSync()[0];

                x = x.dataSync()[idx];
                y = y.dataSync()[idx];
                const stdx = tf.exp(tf.sub(logstdx, bias)).dataSync()[idx];
                const stdy = tf.exp(tf.sub(logstdy, bias)).dataSync()[idx];
                angle = angle.dataSync()[idx];
                e = tf.sigmoid(tf.mul(eos_logit, (1. + 0.*bias))).dataSync()[0];
                const rx = rand_truncated_normal(-5, 5) * stdx;
                const ry = rand_truncated_normal(-5, 5) * stdy;
                x = x + Math.cos(-angle) * rx - Math.sin(-angle) * ry;
                y = y + Math.sin(-angle) * rx + Math.cos(-angle) * ry;
                if (Math.random() < e) {
                    e = 1.;
                } else {
                    e = 0.;
                }
                input = tf.tensor([[x, y, e]]);
                return [att_location, att_context, gru0_hx, gru1_hx, input];
            });

            [dx,dy,eos_] = input.dataSync();
            dy = -dy * 3;
            dx = dx * 3;
            if (eos == 0.) {
                ctx.beginPath();
                ctx.moveTo(cur_x, cur_y, 0, 0);
                ctx.lineTo(cur_x + dx, cur_y + dy);
                ctx.stroke();
            }
            eos = eos_;
            cur_x = cur_x + dx;
            cur_y = cur_y + dy;

            if (att_location.dataSync()[0] < original_text.length + 2) {
                setTimeout(function() {loop(my_run_id);}, 0);
            }
        }
    }

    loop(run_id);
}

window.onload = function(e) {
    //Setting up canvas
    canvas = document.getElementById("hw-canvas");
    ctx = canvas.getContext("2d");
    ctx.canvas.width = window.innerWidth;
    ctx.canvas.height = window.innerHeight;

}
