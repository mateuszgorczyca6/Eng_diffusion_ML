def TAMSD(s, T):
  tamsds = [0] * T
  for n in range(1, T): # gaps
    suma = 0
    for i in range(T - n):
      suma += (s[i+n] - s[i]) ** 2
    tamsds[n] = suma / (T - n)
  return tamsds