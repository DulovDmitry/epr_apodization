import numpy as np 
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

epr_filename = "input.txt"

epr_points = np.loadtxt(epr_filename)
epr_points_field, epr_points_amplitude = epr_points.T

sampling_frequency = 10 # от балды
t = np.arange(0, epr_points_field.size/sampling_frequency, 1/sampling_frequency)

fft_epr_points = fft.ifft(epr_points_amplitude)

t_center = (t[0]+t[-1])/2
t_shift = (t_center - t[0]) / 2
sigma = 1500
apodization_function = np.exp(-(t-t_center-t_shift)**2/sigma)+np.exp(-(t-t_center+t_shift)**2/sigma)
fft_epr_points_transformed = fft_epr_points*apodization_function


'''
Строим графики
'''

fig, ax = plt.subplots()
ax.remove()

plt.subplot(2,2,1)
plt.plot(epr_points_field, epr_points_amplitude, linewidth = 0.8, color='black', label = 'origin spectrum')
plt.legend()

plt.subplot(2,2,2)
line_fft_transformed, = plt.plot(t, fft_epr_points_transformed.real, linewidth = 0.8, color='blue', label = 'epr ifft transformed')
plt.legend()

plt.subplot(2,2,3)
hline = plt.axhline(y=0, color='r', linewidth = 0.5)
line_epr_apodized, = plt.plot(epr_points_field, fft.fft(fft_epr_points_transformed).real, linewidth = 0.8, color='blue', label = 'epr apodized')
plt.legend()

plt.subplot(2,2,4)
line_apodization, = plt.plot(t, apodization_function, linewidth = 0.8, color='blue', label = 'apodization function')
plt.legend()

plot_bottom_margin = 0.25
fig.subplots_adjust(bottom=plot_bottom_margin)


'''
Добавляем ползунки
'''
sliders_left_margin = 0.1
sliders_width = 0.35

ax_t_shift_coarse = fig.add_axes([sliders_left_margin, 0.15, sliders_width, 0.03])
t_shift_coarse_slider = Slider(
    ax=ax_t_shift_coarse,
    label='delta t (coarse)',
    valmin=10,
    valmax=int(t_center),
    valinit=t_shift,
    valstep=1,
)

ax_t_shift_fine = fig.add_axes([sliders_left_margin + sliders_width + 0.1, 0.15, sliders_width, 0.03])
t_shift_fine_slider = Slider(
    ax=ax_t_shift_fine,
    label='delta t (fine)',
    valmin=-10,
    valmax=10,
    valinit=0,
    valstep=0.05,
)

ax_sigma = fig.add_axes([sliders_left_margin, 0.10, sliders_width, 0.03])
sigma_coarse_slider = Slider(
    ax=ax_sigma,
    label="sigma (coarse)",
    valmin=11,
    valmax=1500,
    valinit=sigma,
    valstep=1.0,
)

ax_sigma_fine = fig.add_axes([sliders_left_margin + sliders_width + 0.1, 0.10, sliders_width, 0.03])
sigma_fine_slider = Slider(
    ax=ax_sigma_fine,
    label='sigma (fine)',
    valmin=-10,
    valmax=10,
    valinit=0,
    valstep=0.05,
)

'''
Добавляем кнопки
'''

ax_export_btn = fig.add_axes([sliders_left_margin, 0.02, 0.15, 0.04])
export_btn = Button(
    ax = ax_export_btn,
    label = "export",
)


'''
Добавляем чекбоксы
'''

rax = fig.add_axes([0.125, plot_bottom_margin, 0.04, 0.03])
epr_apodized_plot_checkbox = CheckButtons(
    ax = rax,
    labels = ['y=0'],
    actives = [hline.get_visible()],
    label_props = {'color': 'r'},
    frame_props = {'edgecolor': 'r'},
    check_props = {'facecolor': 'r'},
)


'''
Коллбэки для виджетов
'''

def update_lines_visibility(label):
    hline.set_visible(not hline.get_visible())
    hline.figure.canvas.draw_idle()


def update(val):
    t_shift = t_shift_coarse_slider.val + t_shift_fine_slider.val
    sigma = sigma_coarse_slider.val + sigma_fine_slider.val
    apodization_function = np.exp(-(t-t_center-t_shift)**2/sigma)+np.exp(-(t-t_center+t_shift)**2/sigma)
    fft_epr_points_transformed = fft_epr_points*apodization_function
    epr_apodized = fft.fft(fft_epr_points_transformed)

    line_fft_transformed.set_ydata(fft_epr_points_transformed.real)
    line_epr_apodized.set_ydata(epr_apodized.real)
    line_apodization.set_ydata(apodization_function)
    
    plt.subplot(2,2,2)
    plt.ylim(np.min(fft_epr_points_transformed.real)*1.05, np.max(fft_epr_points_transformed.real)*1.05)

    plt.subplot(2,2,3)
    plt.ylim(np.min(epr_apodized.real)*1.05, np.max(epr_apodized.real)*1.05)

    plt.subplot(2,2,4)
    plt.ylim(np.min(apodization_function)*0.95, np.max(apodization_function)*1.05)


def export_apodized_data(val):
    output_filename = "output.txt"
	
    t_shift = t_shift_coarse_slider.val + t_shift_fine_slider.val
    sigma = sigma_coarse_slider.val + sigma_fine_slider.val
    apodization_function = np.exp(-(t-t_center-t_shift)**2/sigma)+np.exp(-(t-t_center+t_shift)**2/sigma)
    fft_epr_points_transformed = fft_epr_points*apodization_function
    epr_apodized = fft.fft(fft_epr_points_transformed).real

    epr_points_amplitude_norm = epr_points_amplitude/np.max(epr_points_amplitude)
    epr_apodized_norm = epr_apodized/np.max(epr_apodized)

    out_data = np.column_stack(
        (epr_points_field,
         epr_points_amplitude,
         epr_apodized,
         epr_points_amplitude_norm,
         epr_apodized_norm))

    header_str = f'# t_center = {t_center} ; t_shift =  {t_shift}; sigma = {sigma}\n'
    np.savetxt(
        fname = output_filename,
        X = out_data,
        delimiter = '\t',
        header = header_str + 'field\tinitial_spectrum\tapodized_spectrum\tinitial_spectrum_nolmalized\tapodized_spectrum_normalized',
        comments = '')


'''
Связываем виджеты и коллбэки
'''

t_shift_coarse_slider.on_changed(update)
sigma_coarse_slider.on_changed(update)
t_shift_fine_slider.on_changed(update)
sigma_fine_slider.on_changed(update)
export_btn.on_clicked(export_apodized_data)
epr_apodized_plot_checkbox.on_clicked(update_lines_visibility)



plt.show()