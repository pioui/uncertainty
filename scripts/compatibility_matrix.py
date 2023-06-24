import numpy as np

labels = ["16QAM", "64QAM", "8PSK", "B-FM", "BPSK", "CPFSK", "DSB-AM", "GFSK", "PAM4", "QPSK", "SSB-AM"]


def print_latex_matrix(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0

    if num_rows == 0 or num_cols == 0:
        print("Invalid matrix.")
        return

    maxma = np.max(matrix)
    minma = np.min(matrix)

    for i in range(num_rows):
        # row_str = f"{labels[i]} & "
        # row_str = row_str+ " & ".join("{:.2f}".format((elem-minma)/(maxma-minma)) for elem in matrix[i])
        # print(row_str + " \\\\")

        # row_str = " & ".join("{:.2f}".format((elem-minma)/(maxma-minma)) for elem in matrix[i])
        max_value = np.max(matrix[i])
        min_value = np.min(matrix[i][np.nonzero(matrix[i])])

        row_str = " & ".join(("{:.2f}".format(elem*100), "\\mathbf{{{:.2f}}}".format(elem*100))[elem == max_value or elem== min_value] for elem in matrix[i])
        
        # row_str = " & ".join(("\\mathit{{{:.2f}}}".format(elem), "{:.2f}".format(elem), "\\mathbf{{{:.2f}}}".format(elem))[elem == min_value or elem == max_value] for elem in matrix[i])
        

        print(row_str +  f" & \\text{{{labels[i]}}} \\\\")



compatibility_matrix15= np.load('/home/pigi/repos/uncertainty/outputs/signalModulation-SNR-15/signalModulation-SNR-15_omegaH.npy')
compatibility_matrix50= np.load('/home/pigi/repos/uncertainty/outputs/signalModulation-SNR-50/signalModulation-SNR-50_omegaH.npy')

print("omega 15")
print_latex_matrix(compatibility_matrix15)
print("omega 50")
print_latex_matrix(compatibility_matrix50)