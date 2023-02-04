# -*- coding: utf-8 -*-
"""Buying and Selling stocks.ipynb

Automatically generated by Colaboratory.
"""

# Stock Buy and Sell for Maximum Profit
# O(n) time
def maxProfit(prices):
  minPrice = prices[0]
  maxProfit = 0
  for i in range(len(prices)):
    if prices[i] < minPrice:
      minPrice = prices[i]
      minIndex = i
    elif prices[i] - minPrice > maxProfit:
      maxProfit = prices[i] - minPrice
      maxIndex = i
  return maxProfit, minIndex, maxIndex

# O(n**2) time
def maxProf(prices):
  maxProfit = 0
  for i in range(len(prices)):
    for j in range(i+1, len(prices)):
      if prices[j] - prices[i] > maxProfit:
        maxProfit = prices[j] - prices[i] 
        minIndex, maxIndex = i, j
  return maxProfit, minIndex, maxIndex
