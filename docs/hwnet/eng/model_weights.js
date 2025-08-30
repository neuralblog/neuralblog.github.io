var oReq = new XMLHttpRequest();
oReq.open("GET", "weights.bin", true);
oReq.responseType = "arraybuffer";

var weights_meta={'rnn/~/conv1_d__b': [[0, 73], [73]], 'rnn/~/conv1_d__w': [[73, 26718], [5, 73, 73]], 'rnn/~/embed__embeddings': [[26718, 64094], [512, 73]], 'rnn/~/embed_1__embeddings': [[64094, 69423], [73, 73]], 'rnn/~/linear__b': [[69423, 69454], [31]], 'rnn/~/linear__w': [[69454, 85326], [512, 31]], 'rnn/~/lstm_attention_core/~/gru__b': [[85326, 86862], [1536]], 'rnn/~/lstm_attention_core/~/gru__w_h': [[86862, 873294], [512, 1536]], 'rnn/~/lstm_attention_core/~/gru__w_i': [[873294, 990030], [76, 1536]], 'rnn/~/lstm_attention_core/~/gru_1__b': [[990030, 991566], [1536]], 'rnn/~/lstm_attention_core/~/gru_1__w_h': [[991566, 1777998], [512, 1536]], 'rnn/~/lstm_attention_core/~/gru_1__w_i': [[1777998, 2681166], [588, 1536]], 'rnn/~/lstm_attention_core/~/linear__b': [[2681166, 2681169], [3]], 'rnn/~/lstm_attention_core/~/linear__w': [[2681169, 2682933], [588, 3]]};

var WEIGHTS = {};
var weight_buffer = null;
var W = null;
var w32 = null;
var w16 = null;

oReq.onload = function (oEvent) {
  var arrayBuffer = oReq.response; // Note: not oReq.responseText
  if (arrayBuffer) {
    // convert bfloat16 to float32
    // w16 = new Uint16Array(arrayBuffer)
    // weight_buffer = new SharedArrayBuffer(2*arrayBuffer.byteLength);
    // w32 = new Uint16Array(weight_buffer);
    // for(var i=0; i < w16.length; i++) {
    //     w32[i * 2 + 1] = w16[i];
    // }
    W = new Float32Array(arrayBuffer);
    document.getElementById("btn").innerText = "Buffer arrieved";

    for(var k in weights_meta) {
        info = weights_meta[k];
        offset = info[0];
        shape = info[1];
        WEIGHTS[k] = tf.tensor(W.subarray(offset[0], offset[1]), shape);
    }

    document.getElementById("btn").disabled = false;
    tf.engine().startScope();    
    setTimeout(function() {
        cur_run = cur_run + 1;
        dojob(cur_run);
    }, 0);

    document.getElementById("btn").innerText = "Generate";
  }
};

tf.ready().then( function() {
        tf.enableProdMode();
        oReq.send(null);
});
