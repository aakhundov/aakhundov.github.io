const cnv_draw = document.querySelector('#draw');
const cnv_down = document.querySelector('#down');
const cnv_draw_rect = document.querySelector('#draw_rect');
const cnv_recon = document.querySelector('#recon');
const cnv_recon_rect = document.querySelector('#recon_rect');
const cnv_rec_window = [
    document.querySelector('#rec_window_1'),
    document.querySelector('#rec_window_2')
];

const ctx_draw = cnv_draw.getContext('2d');
const ctx_down = cnv_down.getContext('2d');
const ctx_draw_rect = cnv_draw_rect.getContext('2d');
const ctx_recon = cnv_recon.getContext('2d');
const ctx_recon_rect = cnv_recon_rect.getContext('2d');
const ctx_rec_window = [
    cnv_rec_window[0].getContext('2d'),
    cnv_rec_window[1].getContext('2d'),
];

ctx_draw.lineCap = 'round';
ctx_draw.lineJoin = 'round';
ctx_draw.strokeStyle = '#ffffff';
ctx_draw.lineWidth = 14 * cnv_draw.width / 250;
ctx_draw_rect.lineWidth = 2;
ctx_recon_rect.lineWidth = 2;

const rect_colors = [
    '#ff0000', '#00ff00', '#0000ff'
];


let is_drawing = false;
let last_x = 0, last_y = 0;
const touch_map = new Map();

let downsampled_input = null;

let inferred_rects = null;
let inferred_recon = null;
let inferred_rec_windows = null;


function get_downsampled_image(img_data, dim) {
    const result = [];
    const s = Math.round(Math.sqrt(img_data.length / 4));
    const dx = Math.round(s / dim), dy = Math.round(s / dim);

    let last_x = 0, current_x = 0;
    for (let x = 0; x < dim; x++) {
        current_x += dx;
        let last_y = 0, current_y = 0;
        for (let y = 0; y < dim; y++) {
            current_y += dy;

            let sum = 0;
            for (let x = Math.round(last_x); x < Math.round(current_x); x++) {
                for (let y = Math.round(last_y); y < Math.round(current_y); y++) {
                    sum += img_data[4 * (x * s + y)];
                }
            }
            result.push(sum / (dx * dy) / 255);

            last_y = current_y;
        }
        last_x = current_x;
    }

    return result;
}

function render_downsampled_image(ctx, cnv, data) {
    const dim = Math.round(Math.sqrt(data.length));
    const dx = cnv.width / dim, dy = cnv.height / dim;

    ctx.beginPath();

    let last_x = 0, current_x = 0;
    for (let x = 0; x < dim; x++) {
        current_x += dx;
        let last_y = 0, current_y = 0;
        for (let y = 0; y < dim; y++) {
            current_y += dy;

            const color = Math.round(data[y * dim + x] * 255);
            ctx.fillStyle = 'rgb(' + color + ',' + color + ',' + color + ')';
            ctx.fillRect(
                Math.round(last_x), Math.round(last_y),
                Math.round(current_x), Math.round(current_y)
            );

            last_y = current_y;
        }
        last_x = current_x;
    }

    ctx.fill();
}

function draw_rects(ctx, cnv, rects) {
    for (let i = 0; i < rects.length; i++) {
        const [sc, sx, sy] = rects[i];
        const x = (sx + 1) / 2 * cnv.width;
        const y = (sy + 1) / 2 * cnv.height;
        const s = cnv.width * sc;

        ctx.beginPath();
        ctx.strokeStyle = rect_colors[i];
        ctx.rect(x-s/2, y-s/2, s, s);
        ctx.stroke();
    }
}


function update_downsampled_input() {
    downsampled_input = get_downsampled_image(
        ctx_draw.getImageData(0, 0, cnv_draw.width, cnv_draw.height).data, 50
    );
}

function update_downsampled_canvas() {
    render_downsampled_image(ctx_down, cnv_down, downsampled_input);
}

function update_rects() {
    for (let [ctx, cnv] of [[ctx_draw_rect, cnv_draw_rect], [ctx_recon_rect, cnv_recon_rect]]) {
        ctx.clearRect(0, 0, cnv.width, cnv.height);

        if (inferred_rects !== null) {
            draw_rects(ctx, cnv, inferred_rects);
        }
    }
}

function update_reconstruction() {
    ctx_recon.clearRect(0, 0, cnv_recon.width, cnv_recon.height);

    if (inferred_recon !== null) {
        render_downsampled_image(ctx_recon, cnv_recon, inferred_recon);
    }
}

function update_rec_windows() {
    for (let i = 0; i < ctx_rec_window.length; i++) {
        ctx_rec_window[i].clearRect(0, 0, cnv_rec_window[i].width, cnv_rec_window[i].height);

        if (inferred_rec_windows != null && inferred_rec_windows.length > i) {
            render_downsampled_image(ctx_rec_window[i], cnv_rec_window[i], inferred_rec_windows[i]);
        }
    }
}

function update_inference_view() {
    update_rects();
    update_reconstruction();
    update_rec_windows();
}

function clear_all_views() {
    ctx_draw.clearRect(0, 0, cnv_draw.width, cnv_draw.height);

    inferred_rects = null;
    inferred_recon = null;
    inferred_rec_windows = null;

    update_downsampled_input();
    update_downsampled_canvas();
    update_inference_view();
}

function draw_stroke(e) {
    if (!is_drawing) {
        return;
    }

    ctx_draw.beginPath();
    ctx_draw.moveTo(last_x, last_y);
    ctx_draw.lineTo(e.offsetX, e.offsetY);
    ctx_draw.stroke();

    update_downsampled_input();
    update_downsampled_canvas();

    [last_x, last_y] = [e.offsetX, e.offsetY];
}


function touch_start(e) {
    e.preventDefault();

    if (e.touches.length > 1) {
        clear_all_views();
        touch_map.clear();
    }
    else {
        for (let touch of e.changedTouches) {
            touch_map.set(touch.identifier, [touch.pageX, touch.pageY]);
        }
    }
}

function touch_change(e, draw=true, end=false) {
    e.preventDefault();

    for (let touch of e.changedTouches) {
        const id = touch.identifier;

        if (touch_map.has(id)) {
            if (draw) {
                ctx_draw.beginPath();
                ctx_draw.moveTo(...touch_map.get(id));
                ctx_draw.lineTo(touch.pageX, touch.pageY);
                ctx_draw.stroke();

                update_downsampled_input();
                update_downsampled_canvas();
            }

            if (!end) {
                touch_map.set(id, [touch.pageX, touch.pageY]);
            }
            else {
                touch_map.delete(id);
            }
        }
    }
}


cnv_draw.addEventListener('mouseup', () => is_drawing = false );
cnv_draw.addEventListener('mouseout', () => is_drawing = false );
cnv_draw.addEventListener('mousemove', (e) => draw_stroke(e));
cnv_draw.addEventListener('mousedown', (e) => {
    if (e.button === 2) {
        clear_all_views();
    }
    else {
        is_drawing = true;
        [last_x, last_y] = [e.offsetX, e.offsetY];
    }
});

cnv_draw.addEventListener("touchstart", (e) => touch_start(e), false);
cnv_draw.addEventListener("touchend", (e) => touch_change(e, true, true), false);
cnv_draw.addEventListener("touchcancel", (e) => touch_change(e, false, true), false);
cnv_draw.addEventListener("touchmove", (e) => touch_change(e, true, false), false);

clear_all_views();


const math = deeplearn.ENV.math;
const Scalar = deeplearn.Scalar;
const NDArray = deeplearn.NDArray;
const varLoader = new deeplearn.CheckpointLoader('./data');

varLoader.getAllVariables().then(function(vars) {
    // NN components

    function create_lstm(kernel, biases) {
        const forget_bias = Scalar.new(1.0);
        return function(input, last_c, last_h) {
            return math.basicLSTMCell(
                forget_bias, kernel, biases,
                input, last_c, last_h
            );
        }
    }

    function create_dense_layer(weights, biases, activation=((x) => x)) {
        return function(input) {
            const weighted = math.matMul(input, weights);
            const linear = math.add(weighted, biases);
            return activation(linear);
        }
    }

    function chain_dense_layers(...layers) {
        return function (input) {
            let result = input;
            for (let layer of layers) {
                result = layer(result);
            }
            return result;
        }
    }


    // random variables

    function sample_from_bernoulli(log_odds) {
        const odds = math.exp(log_odds);
        const prob = math.divide(odds, math.add(odds, Scalar.new(1.0)));
        const uniform = NDArray.randUniform(prob.shape, 0.0, 1.0);
        return math.less(uniform, prob);
    }

    function sample_from_mv_normal(mean, log_variance) {
        const std = math.sqrt(math.exp(log_variance));
        const normal = NDArray.randNormal(mean.shape);
        return math.add(mean, math.multiply(normal, std));
    }


    // spacial transformer replacement

    function transform_coordinates(rect) {
        const [sc, sx, sy] = rect;
        const dim = Math.round(sc * 50);
        const left = Math.round((sx + 1 - sc) / 2 * 50);
        const top = Math.round((sy + 1 - sc) / 2 * 50);
        return [dim, left, top]
    }

    function extract_patch(input, coords) {
        const [dim, x, y] = coords;
        const input2d = input.as2D(50, 50);
        const padded = math.pad2D(input2d, [[25, 25], [25, 25]]);
        return math.slice2D(padded, [y+25, x+25], [dim, dim]).as1D();
    }

    function resize_image(input, dim) {
        const d = input.shape.length === 1 ?
            Math.round(Math.sqrt(input.shape[0])) :
            input.shape[0];
        return math.resizeBilinear3D(input.as3D(d, d, 1), [dim, dim]).as1D();
    }

    function apply_patch(canvas, patch, coords) {
        const [dim, x, y] = coords;
        const patch2d = patch.as2D(dim, dim);
        const canvas2d = canvas.as2D(50, 50);
        const paddedCanvas = math.pad2D(canvas2d, [[25, 25], [25, 25]]);
        const paddedPatch = math.pad2D(patch2d, [[y + 25, 75 - y - dim], [x + 25, 75 - x - dim]]);
        const summedCanvas = math.clip(math.add(paddedCanvas, paddedPatch), 0.0, 1.0);
        return math.slice2D(summedCanvas, [25, 25], [50, 50]).as1D();
    }


    // common LSTM network

    const lstm = create_lstm(
        vars['air/rnn/rnn/kernel'],
        vars['air/rnn/rnn/bias']
    );

    // pre-sigmoid scale (s) distribution (1D Normal)

    const scale_mean = chain_dense_layers(
        create_dense_layer(
            vars['air/rnn/scale/mean/hidden/weights'],
            vars['air/rnn/scale/mean/hidden/biases'],
            (x) => math.relu(x)
        ),
        create_dense_layer(
            vars['air/rnn/scale/mean/output/weights'],
            vars['air/rnn/scale/mean/output/biases']
        )
    );

    const scale_log_variance = chain_dense_layers(
        create_dense_layer(
            vars['air/rnn/scale/log_variance/hidden/weights'],
            vars['air/rnn/scale/log_variance/hidden/biases'],
            (x) => math.relu(x)
        ),
        create_dense_layer(
            vars['air/rnn/scale/log_variance/output/weights'],
            vars['air/rnn/scale/log_variance/output/biases']
        )
    );

    // pre-tanh shift (x, y) distribution (2D Normal)

    const shift_mean = chain_dense_layers(
        create_dense_layer(
            vars['air/rnn/shift/mean/hidden/weights'],
            vars['air/rnn/shift/mean/hidden/biases'],
            (x) => math.relu(x)
        ),
        create_dense_layer(
            vars['air/rnn/shift/mean/output/weights'],
            vars['air/rnn/shift/mean/output/biases']
        )
    );

    const shift_log_variance = chain_dense_layers(
        create_dense_layer(
            vars['air/rnn/shift/log_variance/hidden/weights'],
            vars['air/rnn/shift/log_variance/hidden/biases'],
            (x) => math.relu(x)
        ),
        create_dense_layer(
            vars['air/rnn/shift/log_variance/output/weights'],
            vars['air/rnn/shift/log_variance/output/biases']
        )
    );

    // z_pres log-odds distribution (Bernoulli)

    const z_pres_log_odds = chain_dense_layers(
        create_dense_layer(
            vars['air/rnn/z_pres/log_odds/hidden/weights'],
            vars['air/rnn/z_pres/log_odds/hidden/biases'],
            (x) => math.relu(x)
        ),
        create_dense_layer(
            vars['air/rnn/z_pres/log_odds/output/weights'],
            vars['air/rnn/z_pres/log_odds/output/biases']
        )
    );

    // variational auto-encoder

    function softplus(x) {
        return math.log(math.add(math.exp(x), Scalar.new(1.0)));
    }

    const vae_rec_network = chain_dense_layers(
        create_dense_layer(
            vars['air/rnn/vae/recognition_1/weights'],
            vars['air/rnn/vae/recognition_1/biases'],
            softplus
        ),
        create_dense_layer(
            vars['air/rnn/vae/recognition_2/weights'],
            vars['air/rnn/vae/recognition_2/biases'],
            softplus
        )
    );

    const vae_rec_mean = create_dense_layer(
        vars['air/rnn/vae/rec_mean/weights'],
        vars['air/rnn/vae/rec_mean/biases']
    );

    const vae_rec_log_variance = create_dense_layer(
        vars['air/rnn/vae/rec_log_variance/weights'],
        vars['air/rnn/vae/rec_log_variance/biases']
    );

    const vae_gen_mean = chain_dense_layers(
        create_dense_layer(
            vars['air/rnn/vae/generative_1/weights'],
            vars['air/rnn/vae/generative_1/biases'],
            softplus
        ),
        create_dense_layer(
            vars['air/rnn/vae/generative_2/weights'],
            vars['air/rnn/vae/generative_2/biases'],
            softplus
        ),
        create_dense_layer(
            vars['air/rnn/vae/gen_mean/weights'],
            vars['air/rnn/vae/gen_mean/biases']
        )
    );

    function vae(x) {
        const vae_rec_net = vae_rec_network(x);
        const vae_latent = sample_from_mv_normal(vae_rec_mean(vae_rec_net), vae_rec_log_variance(vae_rec_net));
        const vae_gen_sample = sample_from_mv_normal(vae_gen_mean(vae_latent), Scalar.new(-2.408));  // std = 0.3
        return math.sigmoid(vae_gen_sample);
    }


    const max_steps = 3;
    const init_c = NDArray.zeros([1, 256]);
    const init_h = NDArray.zeros([1, 256]);
    let prev_input_sum = null, prev_states = [];

    function infer() {
        const input_sum = downsampled_input.reduce((a, b) => a + b, 0);

        if (input_sum === 0) {
            inferred_rects = null;
            inferred_recon = null;
            inferred_rec_windows = null;
        }
        else {
            math.scope((keep) => {
                if (input_sum !== prev_input_sum) {
                    for (let [c, h] of prev_states) {
                        c.dispose();
                        h.dispose();
                    }

                    prev_states = [];
                    prev_input_sum = input_sum;
                }

                const new_rects = [];
                let new_recon = NDArray.zeros([2500]);
                const new_rec_windows = [];

                const input = new NDArray(
                    [1, 2500], deeplearn.float32,
                    downsampled_input
                );

                let c = init_c;
                let h = init_h;

                for (let s = 0; s < max_steps; s++) {
                    if (s >= prev_states.length) {
                        [c, h] = lstm(input, c, h);
                        prev_states.push([keep(c), keep(h)]);
                    }
                    else {
                        [c, h] = prev_states[s];
                    }

                    const z_pres = sample_from_bernoulli(z_pres_log_odds(h));

                    if (z_pres.dataSync()[0] === 0) {
                        break;
                    }

                    const scale = math.sigmoid(sample_from_mv_normal(scale_mean(h), scale_log_variance(h)));
                    const shift = math.tanh(sample_from_mv_normal(shift_mean(h), shift_log_variance(h)));

                    const new_rect = [scale.dataSync()[0], ...shift.dataSync()];
                    const coords = transform_coordinates(new_rect);
                    const patch = extract_patch(input, coords);
                    const window = resize_image(patch, 28);
                    const rec_window = vae(window.as2D(1, window.shape[0]));
                    const rec_patch = resize_image(rec_window.squeeze(), coords[0]);

                    new_rects.push(new_rect);
                    new_recon = apply_patch(new_recon, rec_patch, coords);
                    new_rec_windows.push(rec_window);
                }

                if (new_rects.length > 0) {
                    inferred_rects = new_rects;
                    inferred_recon = new_recon.dataSync();
                    inferred_rec_windows = new_rec_windows.map((w) => w.dataSync());
                }
                else {
                    inferred_rects = null;
                    inferred_recon = null;
                    inferred_rec_windows = null;
                }
            });
        }

        update_inference_view();
    }


    // behind-the-curtain one time heavy inference
    // of randomly generated input for cold start

    downsampled_input = [];
    for (let i = 0; i < 2500; i++) {
        downsampled_input.push(Math.random());
    }

    infer();
    clear_all_views();

    // start the inference loop

    (function inference_loop() {
        new Promise((resolve) => {
            infer();
            resolve();
        }).then(() => {
            setTimeout(inference_loop, 100)
        });
    })();

    // raise the curtain

    $('#loading').remove();
    $('#demo').css('visibility', 'visible');
});
