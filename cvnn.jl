# 𝑳

topology = [-1, 28*28, 2000, 1000, 500, 100, 50, 20, 1]


const USE_QR = false
const USE_DROPOUT = false
const MAX_EPOCH = 1000

function BATCH_SIZE(epoch)
  trunc(epoch / 5) + 1
end



using Gallium
breakpoint_on_error()

const τ,𝔦 = 2π, 1.0im
const 𝔦π, 𝔦τ = 𝔦*π, 𝔦*τ

println("start")

typealias ℂ Complex{Float64}

function Base.angle(z₁::ℂ, z₂::ℂ)::Float64
  α = abs( angle(z₂) - angle(z₁) )
  return min(α, 2π-α)
end

type Layer
  #N::Int            # #neurons
  first::Bool
  last::Bool

  dropout::Float64  # 0.2 means 20% of neurons are dropped/OFF
  ✔︎::Any           # ON/OFF for each neuron

  W::Matrix{ℂ}
  b::Vector{ℂ}

  x::Any
  y::Any
  z::Any

  Δ::Any
  δ::Any


  function Layer(dropout, Nprev, N)
    ✔︎ = ones(N)
    if Nprev==-1
      new(true,false,dropout,✔︎)
    else
      W = rand(N, Nprev) .* exp(𝔦τ*rand(N, Nprev))
      b = rand(N)        .* exp(𝔦τ*rand(N))
      new(false,false,dropout,✔︎,W,b)
    end
  end
end

if false
  Juno.breakpoint(@__FILE__, 158)
end

Pkg.add("MNIST")
using MNIST

function main()
  # NETWORK of LAYERS
  Θ = Vector{Layer}()

  for i = 2 : length(topology)
    dropout = i==2 ? .2 : .5
    L = Layer( USE_DROPOUT ? dropout : 0.0, topology[i-1:i]... )
    push!(Θ, L)
  end
  Θ[end].last = true
  Θ[end].dropout = 0


  𝒳, 𝒴 = traindata()
  # 𝒳 = 𝒳[:,1:6000]
  # 𝒴 = 𝒴[  1:6000]

  for epoch = 1 : MAX_EPOCH
    println("Epoch ", epoch)

    batchSize = BATCH_SIZE(epoch)
    nBatches = trunc(Int, length(𝒴) / batchSize)
    indices = reshape( shuffle(1 : nBatches*batchSize), nBatches, batchSize )

    error = 0.0
    nCorrect = 0

progress( name="batch#" ) do p
    for kBatch = 1 : nBatches

      #print(".")
      progress(p, kBatch/nBatches)

      #Turn units on/off according to dropout
      for L in Θ
        L.✔︎ = rand(length(L.✔︎)) .> L.dropout
        if sum(L.✔︎) == 0
          L.✔︎[1] = true    # ensure ONE unit per layer is ON
        end
      end
      Θ[end].✔︎ = true      # all units in OUTPUT layer ON

      # Get batch!
      𝓍 = 𝒳[:,indices[kBatch,:]] / 255.0
      𝓎 = 𝒴[  indices[kBatch,:]] / 10.0

      Θ[1].y = 𝓍 .* exp( 𝔦π * 𝓍 ) .* Θ[1].✔︎ # -> upper half of unit circle
      T      =      exp( 𝔦τ * 𝓎 )            # -> unit circle

      #Forward propagate x
      for i = 2 : length(Θ)
        ◀︎L, L = Θ[i-1:i]
        L.x = ◀︎L.y
        L.z = L.W * L.x  .+  L.b  # tried / L.W * L.x * (1-L.dropout) but no luck
        # Activation Function: Don't normalize output layer!
        σ(z) = L.last ? z : z ./ norm.(z)
        L.y = σ(L.z) .* L.✔︎
      end

      L = Θ[end]

      # Δ represents ‘network error’ for each neuron in the layer.
      L.Δ = -L.y + T.';

      ∠ = angle.( L.y[1,:], T ) # <-- !!! ASSUMING ONLY ONE OUTPUT NEURON
      nCorrect += sum( ∠ .< π/10 )
      error += mean(∠)



      # Back-propagate network error
      #for ◀︎L in Θ[end-1:-1:1]
      for i = length(Θ) : -1 : 2
        ◀︎L, L = Θ[i-1:i]

        L.δ = L.Δ / (sum(◀︎L.✔︎) + 1)
        if ! ◀︎L.first # no Δ, δ for INPUT layer
          # For each neuron of prev layer ◀︎L, set:
          #    ◀︎L.Δ = SUM over each neuron L in this layer of:
          #        L.δ / L.weightTo◀︎L
          # Examine on paper:
          #    {the vector L.δ} * {the matrix of weight reciprocals 1 ./ L.W}
          δ = L.δ .* L.✔︎
          wᵀ = 1 ./ L.W.'

          ◀︎L.Δ = wᵀ * δ
        end
        #L = ◀︎L
      end

      # For each neuron in network, distribute δ among weights
      for L in Θ[2:end]
        N = size(L.x, 1) # #inputs to final layer
        if USE_QR  &&  L.last  &&  batchSize >= N
          # Optimization, see:
          #    "A modified learning algorithm for the multilayer neural network
          #      with multi-valued neurons based on the complex QR decomposition"
          #    Igor Aizenberg, Antonio Luchetta, and Stefano Manetti
          #    Soft Computing, vol. 16, No 4, April 2012, pp. 563-575
          A = hcat(ones(batchSize), L.x.')
          # Have A δW = L.Δ, want δW
          #   Ax=b  =>  x = A\B  <-- Want this one!
          #   xA=b  =>  x = A/B
          ΔW = A \ L.Δ.'

          𝜕b = ΔW[1]
          𝜕W = ΔW[2:end].'
        else
          penalty = L.last ? 1 :  1 ./ norm.(L.z)
          δ = penalty .* L.δ .* L.✔︎
          x̄ᵀ = L.x'                   # <-- CONJUGATE Transpose!
          𝜕W = δ * x̄ᵀ ./ batchSize
          𝜕b = vec( mean(δ,2) )
        end

        L.W += 𝜕W
        L.b += 𝜕b
      end

    end #kBatch
end
    @printf " Error: %f, Misclassified: %d \n"  (error/nBatches)/(π/2)  nBatches*batchSize-nCorrect
  end #epoch

  println("done!")
end
main()

#ones(Float32,10,5)*im
#fill(one(Float32)*im, 10, 5)
#fill(1f0im, 10, 5)
#ones(Complex{Float32}, 10, 5)

# Note: To avoid allocation, could do:
#   copy!( L_prev.Δ, result )
#   A .= B ??
#   A[:] = B

# L = Dict{Symbol, Any}()  # Dict{Symbol,Any}[]
# L[:W] = exp(ι*τ*rand(N))
# L[:b] = exp(ι*τ*rand())
# L[:N] = N

# Error: 0.334363, Misclassified: 4656 .05     (100 its)
# Error: 0.339458, Misclassified: 4773 0 to 2  (10 its)
# Error: 0.333875, Misclassified: 4669 1       (7 its)
# Error: 0.276041, Misclassified: 4283 0 to 1  (10 its) <-- WINNER
