import math
import torch
import imageio
import numpy as np
from scipy import signal
from operator import pos
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

## TODO (learned prior)
def kl_criterion_lp(mu1, logvar1, mu2, logvar2, args):
	# KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2))
	sigma1 = logvar1.mul(0.5).exp() 
	sigma2 = logvar2.mul(0.5).exp() 
	kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
	return kld.sum() / args.batch_size

def kl_criterion(mu, logvar, args):
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	KLD /= args.batch_size  
	return KLD

def eval_seq(gt, pred):
	T = len(gt)
	bs = gt[0].shape[0]
	# bs = gt[0].shape[0]
	ssim = np.zeros((bs, T))
	psnr = np.zeros((bs, T))
	mse = np.zeros((bs, T))
	for i in range(bs):
		for t in range(T):
			origin = gt[t][i]
			predict = pred[t][i]
			for c in range(origin.shape[0]):
				ssim[i, t] += ssim_metric(origin[c], predict[c]) 
				psnr[i, t] += psnr_metric(origin[c], predict[c])
			ssim[i, t] /= origin.shape[0]
			psnr[i, t] /= origin.shape[0]
			mse[i, t] = mse_metric(origin, predict)

	return mse, ssim, psnr

def mse_metric(x1, x2):
	err = np.sum((x1 - x2) ** 2)
	err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
	return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
	T = len(gt)
	bs = gt[0].shape[0]
	ssim = np.zeros((bs, T))
	psnr = np.zeros((bs, T))
	mse = np.zeros((bs, T))
	for i in range(bs):
		for t in range(T):
			origin = gt[t][i].detach().cpu().numpy()
			predict = pred[t][i].detach().cpu().numpy()
			for c in range(origin.shape[0]):
				res = finn_ssim(origin[c], predict[c]).mean()
				if math.isnan(res):
					ssim[i, t] += -1
				else:
					ssim[i, t] += res
				psnr[i, t] += finn_psnr(origin[c], predict[c])
			ssim[i, t] /= origin.shape[0]
			psnr[i, t] /= origin.shape[0]
			mse[i, t] = mse_metric(origin, predict)

	return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
	mse = ((x - y)**2).mean()
	return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
	x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
	g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
	return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)

	size = 11
	sigma = 1.5
	window = fspecial_gauss(size, sigma)

	K1 = 0.01
	K2 = 0.03

	C1 = (K1 * data_range) ** 2
	C2 = (K2 * data_range) ** 2
	mu1 = signal.fftconvolve(img1, window, mode='valid')
	mu2 = signal.fftconvolve(img2, window, mode='valid')
	mu1_sq = mu1*mu1
	mu2_sq = mu2*mu2
	mu1_mu2 = mu1*mu2
	sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
	sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
	sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

	if cs_map:
		return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
					(sigma1_sq + sigma2_sq + C2)), 
				(2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
	else:
		return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
					(sigma1_sq + sigma2_sq + C2))

def init_weights(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1 or classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

## TODO
def pred(x, cond, modules, args):
	# initialize the hidden state.
	modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
	modules['posterior'].hidden = modules['posterior'].init_hidden()
	gen_seq = []
	gen_seq.append(x[0])
	h_seq = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]
	for i in range(1, args.n_past+args.n_future):
		h_target = h_seq[i][0]
		if args.last_frame_skip or i < args.n_past:	
			h, skip = h_seq[i-1]
		else:
			h, _ = h_seq[i-1]
		h = h.detach()
		if i < args.n_past:
			z_t, _, _ = modules['posterior'](h_target)
			modules['frame_predictor'](torch.cat([h, z_t], 1)) 
			gen_seq.append(x[i])
		else:
			z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
			h_pred = modules['frame_predictor'](torch.cat([h, z_t], 1)).detach()
			x_pred = modules['decoder']([h_pred, skip], cond[i-1]).detach()
			h_seq[i] = modules['encoder'](x_pred)
			gen_seq.append(x_pred)
	return gen_seq

## TODO (learned prior)
def pred_lp(x, cond, modules, args):
	# initialize the hidden state.
	modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
	modules['posterior'].hidden = modules['posterior'].init_hidden()
	modules['prior'].hidden = modules['prior'].init_hidden()
	gen_seq = []
	gen_seq.append(x[0])
	h_seq = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]
	for i in range(1, args.n_past+args.n_future):
		h_target = h_seq[i][0]
		if args.last_frame_skip or i < args.n_past:	
			h, skip = h_seq[i-1]
		else:
			h, _ = h_seq[i-1]
		h = h.detach()
		if i < args.n_past:
			z_t, _, _ = modules['posterior'](h_target)
			modules['frame_predictor'](torch.cat([h, z_t], 1)) 
			gen_seq.append(x[i])
		else:
			z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
			h_pred = modules['frame_predictor'](torch.cat([h, z_t], 1)).detach()
			x_pred = modules['decoder']([h_pred, skip], cond[i-1]).detach()
			h_seq[i] = modules['encoder'](x_pred)
			gen_seq.append(x_pred)
	return gen_seq

## ----------------- scipy.misc.toimage ------------------##
def bytescale(data, cmin=None, cmax=None, high=255, low=0):
	if data.dtype == np.uint8:
		return data

	if high > 255:
		raise ValueError("`high` should be less than or equal to 255.")
	if low < 0:
		raise ValueError("`low` should be greater than or equal to 0.")
	if high < low:
		raise ValueError("`high` should be greater than or equal to `low`.")

	if cmin is None:
		cmin = data.min()
	if cmax is None:
		cmax = data.max()

	cscale = cmax - cmin
	if cscale < 0:
		raise ValueError("`cmax` should be larger than `cmin`.")
	elif cscale == 0:
		cscale = 1

	scale = float(high - low) / cscale
	bytedata = (data - cmin) * scale + low
	return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None, mode=None, channel_axis=None):
	data = np.asarray(arr)
	if np.iscomplexobj(data):
		raise ValueError("Cannot convert a complex-valued array.")
	shape = list(data.shape)
	valid = len(shape) == 2 or ((len(shape) == 3) and
								((3 in shape) or (4 in shape)))
	if not valid:
		raise ValueError("'arr' does not have a suitable array shape for "
						 "any mode.")
	if len(shape) == 2:
		shape = (shape[1], shape[0])  # columns show up first
		if mode == 'F':
			data32 = data.astype(np.float32)
			image = Image.frombytes(mode, shape, data32.tostring())
			return image
		if mode in [None, 'L', 'P']:
			bytedata = bytescale(data, high=high, low=low,
								 cmin=cmin, cmax=cmax)
			image = Image.frombytes('L', shape, bytedata.tostring())
			if pal is not None:
				image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
				# Becomes a mode='P' automagically.
			elif mode == 'P':  # default gray-scale
				pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
					   np.ones((3,), dtype=np.uint8)[np.newaxis, :])
				image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
			return image
		if mode == '1':  # high input gives threshold for 1
			bytedata = (data > high)
			image = Image.frombytes('1', shape, bytedata.tostring())
			return image
		if cmin is None:
			cmin = np.amin(np.ravel(data))
		if cmax is None:
			cmax = np.amax(np.ravel(data))
		data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
		if mode == 'I':
			data32 = data.astype(np.uint32)
			image = Image.frombytes(mode, shape, data32.tostring())
		else:
			raise ValueError(_errstr)
		return image

	# if here then 3-d array with a 3 or a 4 in the shape length.
	# Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
	if channel_axis is None:
		if (3 in shape):
			ca = np.flatnonzero(np.asarray(shape) == 3)[0]
		else:
			ca = np.flatnonzero(np.asarray(shape) == 4)
			if len(ca):
				ca = ca[0]
			else:
				raise ValueError("Could not find channel dimension.")
	else:
		ca = channel_axis

	numch = shape[ca]
	if numch not in [3, 4]:
		raise ValueError("Channel axis dimension is not valid.")

	bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
	if ca == 2:
		strdata = bytedata.tostring()
		shape = (shape[1], shape[0])
	elif ca == 1:
		strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
		shape = (shape[2], shape[0])
	elif ca == 0:
		strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
		shape = (shape[2], shape[1])
	if mode is None:
		if numch == 3:
			mode = 'RGB'
		else:
			mode = 'RGBA'

	if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
		raise ValueError(_errstr)

	if mode in ['RGB', 'YCbCr']:
		if numch != 3:
			raise ValueError("Invalid array shape for mode.")
	if mode in ['RGBA', 'CMYK']:
		if numch != 4:
			raise ValueError("Invalid array shape for mode.")

	# Here we know data and mode is correct
	image = Image.frombytes(mode, shape, strdata)
	return image
## ----------------- scipy.misc.toimage ------------------##

## ----------------------- TOOL --------------------------##
def add_border(x, color, pad=1):
	w = x.size()[1]
	nc = x.size()[0]
	px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
	if color == 'red':
		px[0] =0.7 
	elif color == 'green':
		px[1] = 0.7
	if nc == 1:
		for c in range(3):
			px[c, pad:w+pad, pad:w+pad] = x
	else:
		px[:, pad:w+pad, pad:w+pad] = x
	return px

def image_tensor(inputs, padding=1):
	assert len(inputs) > 0
	# print(inputs)

	# if this is a list of lists, unpack them all and grid them up
	if ((not hasattr(inputs[0], "strip") and not type(inputs[0]) is np.ndarray and not hasattr(inputs[0], "dot") and (hasattr(inputs[0], "__getitem__") or hasattr(inputs[0], "__iter__"))) 
		or (hasattr(inputs, "dim") and inputs.dim() > 4)):
			images = [image_tensor(x) for x in inputs]
			if images[0].dim() == 3:
				c_dim = images[0].size(0)
				x_dim = images[0].size(1)
				y_dim = images[0].size(2)
			else:
				c_dim = 1
				x_dim = images[0].size(0)
				y_dim = images[0].size(1)

			result = torch.ones(c_dim, x_dim * len(images) + padding * (len(images)-1), y_dim)
			for i, image in enumerate(images):
				result[:, i * x_dim + i * padding : (i+1) * x_dim + i * padding, :].copy_(image)

			return result

	# if this is just a list, make a stacked image
	else:
		images = [x.data if isinstance(x, torch.autograd.Variable) else x
					for x in inputs]
		# print(images)
		if images[0].dim() == 3:
			c_dim = images[0].size(0)
			x_dim = images[0].size(1)
			y_dim = images[0].size(2)
		else:
			c_dim = 1
			x_dim = images[0].size(0)
			y_dim = images[0].size(1)

		result = torch.ones(c_dim, x_dim, y_dim * len(images) + padding * (len(images)-1))
		for i, image in enumerate(images):
			result[:, :, i * y_dim + i * padding : (i+1) * y_dim + i * padding].copy_(image)
		return result

def draw_text_tensor(tensor, text):
	np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
	pil = Image.fromarray(np.uint8(np_x*255))
	draw = ImageDraw.Draw(pil)
	draw.text((4, 64), text, (0,0,0))
	img = np.asarray(pil)
	return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def save_gif(filename, inputs, duration=0.75):
	images = []
	for tensor in inputs:
		img = image_tensor(tensor, padding=0)
		img = img.cpu()
		img = img.transpose(0,1).transpose(1,2).clamp(0,1)
		images.append(img.numpy())
	imageio.mimsave(filename, img_as_ubyte(images), duration=duration, loop=0)

def save_gif_with_text(filename, inputs, text, duration=0.75):
	images = []
	for tensor, text in zip(inputs, text):
		img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
		img = img.cpu()
		img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
		images.append(img)
	imageio.mimsave(filename, img_as_ubyte(images), duration=duration, loop=0)

def save_tensors_image(filename, inputs, padding=1):
	tensor = image_tensor(inputs, padding)
	tensor = tensor.cpu().clamp(0, 1)
	if tensor.size(0) == 1:
		tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
	# pdb.set_trace()
	tnp = tensor.numpy()
	img = toimage(tnp, high=255*tnp.max(), channel_axis=0)
	return img.save(filename)
## ----------------------- TOOL --------------------------##


## TODO (plot)
# plot the predict result
def plot_pred(x, cond, modules, epoch, args):
	nsample = 5 
	gen_seq = [[] for _ in range(nsample)]
	gt_seq = [x[i] for i in range(len(x))]
	h_seq = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]
	for s in range(nsample):
		modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
		gen_seq[s].append(x[0])
		for i in range(1, args.n_past+args.n_future):
			h_target = h_seq[i][0].detach()
			if args.last_frame_skip or i < args.n_past:
				h, skip = h_seq[i-1]
			else:
				h, _ = h_seq[i-1]
			h = h.detach()
			if i < args.n_past:
				z_t, _, _ = modules['posterior'](h_target)
				modules['frame_predictor'](torch.cat([h, z_t], 1))
				gen_seq[s].append(x[i])
			else:
				z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
				h_pred = modules['frame_predictor'](torch.cat([h, z_t], 1)).detach()
				x_pred = modules['decoder']([h_pred, skip], cond[i-1]).detach()
				h_seq[i] = modules['encoder'](x_pred)
				gen_seq[s].append(x_pred)

	to_plot = []
	gifs = [[] for _ in range(args.n_past+args.n_future)]
	nrow = min(args.batch_size, 10)
	for i in range(nrow):
		# ground truth sequence
		row = [] 
		for t in range(args.n_past+args.n_future):
			row.append(gt_seq[t][i])
		to_plot.append(row)

		for s in range(nsample):
			row = []
			for t in range(args.n_past+args.n_future):
				row.append(gen_seq[s][t][i])
			to_plot.append(row)
		for t in range(args.n_past+args.n_future):
			row = []
			row.append(gt_seq[t][i])
			for s in range(nsample):
				row.append(gen_seq[s][t][i])
			gifs[t].append(row)

	fname = '%s/gen/sample_%d.png' % (args.log_dir, epoch) 
	save_tensors_image(fname, to_plot)

	fname = '%s/gen/sample_%d.gif' % (args.log_dir, epoch) 
	save_gif(fname, gifs)

# plot the reconstruction result
def plot_rec(x, cond, modules, epoch, args):
	modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
	modules['posterior'].hidden = modules['posterior'].init_hidden()
	gen_seq = []
	gen_seq.append(x[0])
	h_seq = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]
	for i in range(1, args.n_past+args.n_future):
		h_target = h_seq[i][0].detach()
		if args.last_frame_skip or i < args.n_past:	
			h, skip = h_seq[i-1]
		else:
			h, _ = h_seq[i-1]
		h = h.detach()
		z_t, _, _ = modules['posterior'](h_target)
		if i < args.n_past:
			modules['frame_predictor'](torch.cat([h, z_t], 1)) 
			gen_seq.append(x[i])
		else:
			h_pred = modules['frame_predictor'](torch.cat([h, z_t], 1)).detach()
			x_pred = modules['decoder']([h_pred, skip], cond[i-1]).detach()
			gen_seq.append(x_pred)
	 
	to_plot = []
	nrow = min(args.batch_size, 10)
	for i in range(nrow):
		row = []
		for t in range(args.n_past+args.n_future):
			row.append(gen_seq[t][i]) 
		to_plot.append(row)

	fname = '%s/gen/rec_%d.png' % (args.log_dir, epoch) 
	save_tensors_image(fname, to_plot)

# plot the demo result (gif)
def plot_demo(x, cond, modules, args):
	# get approx posterior sample
	modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
	modules['posterior'].hidden = modules['posterior'].init_hidden()
	posterior_gen = []
	posterior_gen.append(x[0])
	h_seq = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]
	for i in range(1, args.n_past+args.n_future):
		h_target = h_seq[i][0].detach()
		if args.last_frame_skip or i < args.n_past:	
			h, skip = h_seq[i-1]
		else:
			h, _ = h_seq[i-1]
		h = h.detach()
		_, z_t, _= modules['posterior'](h_target) # take the mean
		if i < args.n_past:
			modules['frame_predictor'](torch.cat([h, z_t], 1)) 
			posterior_gen.append(x[i])
		else:
			h_pred = modules['frame_predictor'](torch.cat([h, z_t], 1)).detach()
			x_pred = modules['decoder']([h_pred, skip], cond[i-1]).detach()
			h_seq[i] = modules['encoder'](x_pred)
			posterior_gen.append(x_pred)
  
	nsample = 3
	psnr = np.zeros((args.batch_size, nsample, args.n_future))
	
	all_gen = []
	for s in range(nsample):
		gen_seq = []
		gt_seq = []
		modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
		modules['posterior'].hidden = modules['posterior'].init_hidden()
		all_gen.append([])
		all_gen[s].append(x[0])
		h_seq = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]
		for i in range(1, args.n_past+args.n_future):
			h_target = h_seq[i][0].detach()
			if args.last_frame_skip or i < args.n_past:	
				h, skip = h_seq[i-1]
			else:
				h, _ = h_seq[i-1]
			h = h.detach()
			if i < args.n_past:
				h_target = modules['encoder'](x[i])[0].detach()
				_, z_t, _ = modules['posterior'](h_target)
			else:
				z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
			if i < args.n_past:
				modules['frame_predictor'](torch.cat([h, z_t], 1))
				all_gen[s].append(x[i])
			else:
				h_pred = modules['frame_predictor'](torch.cat([h, z_t], 1)).detach()
				x_pred = modules['decoder']([h_pred, skip], cond[i-1]).detach()
				gen_seq.append(x_pred)
				gt_seq.append(x[i])
				all_gen[s].append(x_pred)
		_, _, psnr[:, s, :] = finn_eval_seq(gt_seq, gen_seq)

	###### psnr ######
	for i in range(args.batch_size):
		gifs = [ [] for t in range(args.n_past + args.n_future) ]
		text = [ [] for t in range(args.n_past + args.n_future) ]
		mean_psnr = np.mean(psnr[i], 1)
		ordered = np.argsort(mean_psnr)
		rand_sidx = [np.random.randint(nsample) for s in range(3)]
		for t in range(args.n_past + args.n_future):
			# gt 
			gifs[t].append(add_border(x[t][i], 'green'))
			text[t].append('Ground\ntruth')
			#posterior 
			if t < args.n_past:
				color = 'green'
			else:
				color = 'red'
			gifs[t].append(add_border(posterior_gen[t][i], color))
			text[t].append('Approx.\nposterior')
			# best 
			if t < args.n_past:
				color = 'green'
			else:
				color = 'red'
			sidx = ordered[-1]
			gifs[t].append(add_border(all_gen[sidx][t][i], color))
			text[t].append('Best PSNR')
			# random 3
			for s in range(len(rand_sidx)):
				gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
				text[t].append('Random\nsample %d' % (s+1))

		fname = 'best_model/gen/demo.gif'
		save_gif_with_text(fname, gifs, text)



