# SPDX-License-Identifier: MIT OR Apache-2.0

using TemporalFocus
using Test
using Random

@testset "TemporalFocus" begin
    @testset "Discrete Attention" begin
        q = SpikeTrain([
            SpikeEvent(1, 0.10f0, 1.0f0),
            SpikeEvent(2, 0.20f0, 1.0f0),
        ])
        k = SpikeTrain([
            SpikeEvent(1, 0.15f0, 1.0f0),
            SpikeEvent(1, 0.25f0, 1.0f0),
            SpikeEvent(3, 0.30f0, 1.0f0),
        ])
        v = Float32[
            1 0
            0 1
            1 1
        ]

        out = spike_attention_discrete(q, k, v)

        @test out == Float32[2, 0]
    end

    @testset "Temporal Attention" begin
        q = SpikeTrain([SpikeEvent(1, 0.10f0, 1.0f0)])
        k = SpikeTrain([SpikeEvent(1, 0.30f0, 1.0f0)])
        v = Float32[
            2 1
            0 3
        ]

        out = spike_attention_temporal(q, k, v; τ = 0.20f0)
        expected_weight = exp(-1.0f0)

        @test out ≈ expected_weight .* Float32[2, 1] atol = 1.0f-6
    end

    @testset "Continuous Attention" begin
        buffer_q = TemporalBuffer(0.30f0, [SpikeEvent(1, 0.50f0, 1.0f0)])
        buffer_k = TemporalBuffer(0.30f0, [
            SpikeEvent(1, 0.35f0, 1.0f0),
            SpikeEvent(1, 0.90f0, 1.0f0),
        ])
        v = Float32[
            1 2
            3 4
        ]

        out = spike_attention_continuous(buffer_q, buffer_k, v; τ = 0.15f0)

        @test out ≈ exp(-1.0f0) .* Float32[1, 2] atol = 1.0f-6
    end

    @testset "Normalization" begin
        l1 = Float32[2, 2, 4]
        maxn = Float32[2, 6, 3]

        @test normalize_l1!(l1) == Float32[0.25, 0.25, 0.5]
        @test normalize_max!(maxn) == Float32[1 / 3, 1, 0.5]
    end

    @testset "Buffer Pruning" begin
        @testset "Basic pruning" begin
            buffer = TemporalBuffer(0.25f0, [
                SpikeEvent(1, 0.10f0, 1.0f0),
                SpikeEvent(2, 0.55f0, 1.0f0),
                SpikeEvent(3, 0.80f0, 1.0f0),
            ])

            prune!(buffer, 0.80f0)

            @test length(buffer.events) == 2
            @test [event.neuron_id for event in buffer.events] == [2, 3]
        end

        @testset "Prune empty buffer" begin
            buffer = TemporalBuffer(0.25f0, SpikeEvent[])
            prune!(buffer, 1.0f0)
            @test isempty(buffer.events)
        end

        @testset "Prune removes all events" begin
            buffer = TemporalBuffer(0.10f0, [
                SpikeEvent(1, 0.10f0, 1.0f0),
                SpikeEvent(2, 0.20f0, 1.0f0),
            ])
            prune!(buffer, 10.0f0)
            @test isempty(buffer.events)
        end

        @testset "Prune keeps all events" begin
            events = [SpikeEvent(1, 0.90f0, 1.0f0), SpikeEvent(2, 0.95f0, 1.0f0)]
            buffer = TemporalBuffer(1.0f0, events)
            prune!(buffer, 1.0f0)
            @test length(buffer.events) == 2
        end
    end

    @testset "SpikeEvent constructor" begin
        @test SpikeEvent(1, 0.5).t isa Float32
        @test SpikeEvent(1, 0.5).value == 1.0f0
        @test SpikeEvent(1, 0.5, 2.0).value == 2.0f0
    end

    @testset "SpikeTrain constructor" begin
        @test isempty(SpikeTrain().events)
        @test SpikeTrain([SpikeEvent(1, 0.5f0)]).events[1].neuron_id == 1
    end

    @testset "TemporalBuffer constructor" begin
        buf = TemporalBuffer(1.0f0)
        @test buf.window == 1.0f0
        @test isempty(buf.events)
    end

    @testset "Discrete edge cases" begin
        @testset "Empty spike trains" begin
            q = SpikeTrain()
            k = SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)])
            v = Float32[1 0; 0 1]
            out = spike_attention_discrete(q, k, v)
            @test out == Float32[0, 0]
        end

        @testset "Single neuron multiple coincidences" begin
            q = SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)])
            k = SpikeTrain([SpikeEvent(1, 0.2f0, 1.0f0), SpikeEvent(1, 0.3f0, 1.0f0)])
            v = Float32[1; 2;;]
            out = spike_attention_discrete(q, k, v)
            @test out == Float32[2]
        end

        @testset "Non-unit spike values" begin
            q = SpikeTrain([SpikeEvent(1, 0.1f0, 2.0f0)])
            k = SpikeTrain([SpikeEvent(1, 0.2f0, 3.0f0)])
            v = Float32[1; 2;;]
            out = spike_attention_discrete(q, k, v)
            @test out == Float32[6]
        end

        @testset "Out of range source neuron ID throws" begin
            q = SpikeTrain([SpikeEvent(5, 0.1f0, 1.0f0)])
            k = SpikeTrain([SpikeEvent(1, 0.2f0, 1.0f0)])
            v = Float32[1 0; 0 1]
            @test_throws ArgumentError spike_attention_discrete(q, k, v)
        end

        @testset "Out of range context neuron ID is ignored" begin
            q = SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)])
            k = SpikeTrain([SpikeEvent(5, 0.2f0, 1.0f0)])
            v = Float32[1 0; 0 1]
            out = spike_attention_discrete(q, k, v)
            @test out == Float32[0, 0]
        end

        @testset "Larger readout matrix works" begin
            q = SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)])
            k = SpikeTrain([SpikeEvent(1, 0.2f0, 1.0f0)])
            v = Float32[1 0 0; 0 1 0; 0 0 1]
            # readout has 3 rows but only 1 neuron → should work
            out = spike_attention_discrete(q, k, v)
            @test length(out) == 3
        end

        @testset "Dimension mismatch throws" begin
            @test_throws DimensionMismatch TemporalFocus._apply_readout(Float32[1, 2], Float32[1 0 0; 0 1 0; 0 0 1])
        end

        @testset "No coincidences" begin
            q = SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)])
            k = SpikeTrain([SpikeEvent(2, 0.2f0, 1.0f0)])
            v = Float32[1 0; 0 1]
            out = spike_attention_discrete(q, k, v)
            @test out == Float32[0, 0]
        end
    end

    @testset "Temporal edge cases" begin
        @testset "tau zero throws" begin
            q = SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)])
            k = SpikeTrain([SpikeEvent(1, 0.2f0, 1.0f0)])
            v = Float32[1; 2;;]
            @test_throws ArgumentError spike_attention_temporal(q, k, v; τ = 0.0f0)
        end

        @testset "tau zero throws even without coincidences" begin
            # τ is validated eagerly before the loop
            q = SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)])
            k = SpikeTrain([SpikeEvent(2, 0.2f0, 1.0f0)])
            v = Float32[1 0; 0 1]
            @test_throws ArgumentError spike_attention_temporal(q, k, v; τ = 0.0f0)
        end

        @testset "Negative tau throws" begin
            q = SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)])
            k = SpikeTrain([SpikeEvent(1, 0.2f0, 1.0f0)])
            v = Float32[1; 2;;]
            @test_throws ArgumentError spike_attention_temporal(q, k, v; τ = -1.0f0)
        end

        @testset "Small tau decays faster" begin
            q = SpikeTrain([SpikeEvent(1, 0.10f0, 1.0f0)])
            k = SpikeTrain([SpikeEvent(1, 0.30f0, 1.0f0)])
            v = Float32[1; 2;;]
            out_small = spike_attention_temporal(q, k, v; τ = 0.05f0)
            out_large = spike_attention_temporal(q, k, v; τ = 1.0f0)
            @test abs(out_small[1]) < abs(out_large[1])
        end

        @testset "Empty context returns zero" begin
            q = SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)])
            k = SpikeTrain()
            v = Float32[1; 2;;]
            out = spike_attention_temporal(q, k, v; τ = 1.0f0)
            @test out == Float32[0]
        end

        @testset "No coincidences returns zero" begin
            q = SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)])
            k = SpikeTrain([SpikeEvent(2, 0.2f0, 1.0f0)])
            v = Float32[1 0; 0 1]
            out = spike_attention_temporal(q, k, v; τ = 1.0f0)
            @test out == Float32[0, 0]
        end
    end

    @testset "Continuous edge cases" begin
        @testset "Empty buffers" begin
            bq = TemporalBuffer(0.3f0, SpikeEvent[])
            bk = TemporalBuffer(0.3f0, [SpikeEvent(1, 0.5f0, 1.0f0)])
            v = Float32[1; 2;;]
            out = spike_attention_continuous(bq, bk, v; τ = 0.15f0)
            @test out == Float32[0]
        end

        @testset "Events outside window" begin
            bq = TemporalBuffer(0.1f0, [SpikeEvent(1, 0.5f0, 1.0f0)])
            bk = TemporalBuffer(0.1f0, [SpikeEvent(1, 10.0f0, 1.0f0)])
            v = Float32[1; 2;;]
            out = spike_attention_continuous(bq, bk, v; τ = 0.15f0)
            @test out == Float32[0]
        end

        @testset "Different window sizes" begin
            bq = TemporalBuffer(0.5f0, [SpikeEvent(1, 0.5f0, 1.0f0)])
            bk = TemporalBuffer(0.1f0, [SpikeEvent(1, 0.55f0, 1.0f0)])
            v = Float32[1; 2;;]
            out = spike_attention_continuous(bq, bk, v; τ = 0.15f0)
            @test out[1] > 0
        end

        @testset "tau zero throws" begin
            bq = TemporalBuffer(0.3f0, [SpikeEvent(1, 0.5f0, 1.0f0)])
            bk = TemporalBuffer(0.3f0, [SpikeEvent(1, 0.5f0, 1.0f0)])
            v = Float32[1; 2;;]
            @test_throws ArgumentError spike_attention_continuous(bq, bk, v; τ = 0.0f0)
        end

        @testset "tau zero throws even without coincidences" begin
            # Different neuron_ids → no coincidences, but τ is still validated eagerly
            bq = TemporalBuffer(0.001f0, [SpikeEvent(1, 0.5f0, 1.0f0)])
            bk = TemporalBuffer(0.001f0, [SpikeEvent(2, 0.5f0, 1.0f0)])
            v = Float32[1 0; 0 1]
            @test_throws ArgumentError spike_attention_continuous(bq, bk, v; τ = 0.0f0)
        end
    end

    @testset "Normalization edge cases" begin
        @testset "All zeros" begin
            l1 = Float32[0, 0, 0]
            @test normalize_l1!(l1) == Float32[0, 0, 0]

            maxn = Float32[0, 0, 0]
            @test normalize_max!(maxn) == Float32[0, 0, 0]
        end

        @testset "Single element" begin
            l1 = Float32[5]
            @test normalize_l1!(l1) == Float32[1]

            maxn = Float32[5]
            @test normalize_max!(maxn) == Float32[1]
        end

        @testset "Already normalized" begin
            l1 = Float32[0.5, 0.5]
            @test normalize_l1!(l1) ≈ Float32[0.5, 0.5]

            maxn = Float32[0.5, 1.0]
            @test normalize_max!(maxn) ≈ Float32[0.5, 1.0]
        end

        @testset "Negative weights" begin
            # sum is 0, normalize_l1! returns unchanged
            l1 = Float32[-2, 3, -1]
            result = normalize_l1!(l1)
            @test result == Float32[-2, 3, -1]

            # sum is positive, normalizes
            l2 = Float32[-1, 3, 2]
            normalize_l1!(l2)
            @test sum(l2) ≈ 1.0f0
        end

        @testset "Idempotency" begin
            l1 = Float32[2, 2, 4]
            normalize_l1!(l1)
            l1_copy = copy(l1)
            normalize_l1!(l1)
            @test l1 ≈ l1_copy

            maxn = Float32[2, 6, 3]
            normalize_max!(maxn)
            maxn_copy = copy(maxn)
            normalize_max!(maxn)
            @test maxn ≈ maxn_copy
        end
    end

    @testset "Temporal weight" begin
        @testset "Symmetry" begin
            τ = 0.5f0
            @test temporal_weight(0.3f0, τ) ≈ temporal_weight(-0.3f0, τ)
        end

        @testset "Zero dt" begin
            @test temporal_weight(0.0f0, 1.0f0) ≈ 1.0f0
        end

        @testset "Large dt decays" begin
            @test temporal_weight(10.0f0, 1.0f0) < 0.001f0
        end

        @testset "Monotonic decay" begin
            τ = 1.0f0
            @test temporal_weight(0.1f0, τ) > temporal_weight(0.5f0, τ) > temporal_weight(1.0f0, τ)
        end

        @testset "Unchecked matches public when τ > 0" begin
            @test TemporalFocus._temporal_weight_unchecked(0.3f0, 0.5f0) ≈ temporal_weight(0.3f0, 0.5f0)
        end

        @testset "Public still validates τ" begin
            @test_throws ArgumentError temporal_weight(0.1f0, 0.0f0)
            @test_throws ArgumentError temporal_weight(0.1f0, -1.0f0)
        end
    end


    @testset "Base.isempty" begin
        @test isempty(SpikeTrain())
        @test !isempty(SpikeTrain([SpikeEvent(1, 0.1f0, 1.0f0)]))
        @test isempty(TemporalBuffer(1.0f0))
        @test !isempty(TemporalBuffer(1.0f0, [SpikeEvent(1, 0.1f0, 1.0f0)]))
    end


    @testset "Base.show" begin
        e = SpikeEvent(2, 0.5f0, 1.25f0)
        s = sprint(show, e)
        @test occursin("neuron_id=2", s)
        @test occursin("value=", s)
        @test sprint(show, SpikeTrain()) == "SpikeTrain(0 events)"
        @test sprint(show, SpikeTrain([SpikeEvent(1, 0.1f0)])) == "SpikeTrain(1 event)"
        @test sprint(show, SpikeTrain([SpikeEvent(1, 0.1f0), SpikeEvent(2, 0.2f0)])) == "SpikeTrain(2 events)"
        @test occursin("TemporalBuffer(window=", sprint(show, TemporalBuffer(0.25f0)))
        @test occursin("0 events)", sprint(show, TemporalBuffer(0.25f0)))
        @test occursin("1 event)", sprint(show, TemporalBuffer(0.25f0, [SpikeEvent(1, 0.1f0)])))
    end


    @testset "Base.==" begin
        @testset "SpikeEvent" begin
            a = SpikeEvent(1, 0.5f0, 1.0f0)
            @test a == SpikeEvent(1, 0.5f0, 1.0f0)
            @test a != SpikeEvent(2, 0.5f0, 1.0f0)
            @test a != SpikeEvent(1, 0.6f0, 1.0f0)
            @test a != SpikeEvent(1, 0.5f0, 2.0f0)
            # ±0.0f0: Float32 `==` collapses signs; `isequal`/`Set` keep Julia float rules
            zpos = SpikeEvent(1, 0.0f0, 0.0f0)
            zneg = SpikeEvent(1, -0.0f0, -0.0f0)
            @test zpos == zneg
            @test !isequal(zpos, zneg)
            @test length(Set([zpos, zneg])) == 2
            # NaN: `==` is false; `isequal`/`Set` treat matching NaN spikes as one key
            nan1 = SpikeEvent(1, NaN32, 1.0f0)
            nan2 = SpikeEvent(1, NaN32, 1.0f0)
            @test nan1 != nan2
            @test isequal(nan1, nan2)
            @test hash(nan1) == hash(nan2)
            @test length(Set([nan1, nan2])) == 1
        end
        @testset "SpikeTrain" begin
            e1 = SpikeEvent(1, 0.1f0, 1.0f0)
            e2 = SpikeEvent(2, 0.2f0, 1.0f0)
            @test SpikeTrain() == SpikeTrain(SpikeEvent[])
            @test SpikeTrain([e1, e2]) == SpikeTrain([e1, e2])
            @test SpikeTrain([e1, e2]) != SpikeTrain([e2, e1])
            @test SpikeTrain([e1]) != SpikeTrain()
            @test hash(SpikeTrain([e1, e2])) == hash(SpikeTrain([e1, e2]))
            @test length(Set([SpikeTrain([e1]), SpikeTrain([e1])])) == 1
            tnan1 = SpikeTrain([SpikeEvent(1, NaN32, 1.0f0)])
            tnan2 = SpikeTrain([SpikeEvent(1, NaN32, 1.0f0)])
            @test tnan1 != tnan2
            @test isequal(tnan1, tnan2)
            @test length(Set([tnan1, tnan2])) == 1
            # Mutable-key contract: content hash changes if events mutate while
            # the object is used as a Dict/Set key (do not rely on one specific
            # corrupted-lookup outcome — that is hash-table implementation-defined).
            key = SpikeTrain([e1])
            h_before = hash(key)
            push!(key.events, e2)
            @test hash(key) != h_before
        end
        @testset "TemporalBuffer" begin
            e = SpikeEvent(1, 0.1f0, 1.0f0)
            @test TemporalBuffer(1.0f0) == TemporalBuffer(1.0f0, SpikeEvent[])
            @test TemporalBuffer(1.0f0, [e]) == TemporalBuffer(1.0f0, [e])
            @test TemporalBuffer(1.0f0, [e]) != TemporalBuffer(2.0f0, [e])
            @test TemporalBuffer(1.0f0, [e]) != TemporalBuffer(1.0f0)
            @test hash(TemporalBuffer(1.0f0, [e])) == hash(TemporalBuffer(1.0f0, [e]))
            bpos = TemporalBuffer(0.0f0)
            bneg = TemporalBuffer(-0.0f0)
            @test bpos == bneg
            @test !isequal(bpos, bneg)
            @test length(Set([bpos, bneg])) == 2
            bnan1 = TemporalBuffer(NaN32)
            bnan2 = TemporalBuffer(NaN32)
            @test bnan1 != bnan2
            @test isequal(bnan1, bnan2)
            @test length(Set([bnan1, bnan2])) == 1
            # Mutable-key contract: prune! changes content hash (same caveat as SpikeTrain)
            buf = TemporalBuffer(0.5f0, [SpikeEvent(1, 0.0f0, 1.0f0), SpikeEvent(1, 1.0f0, 1.0f0)])
            h_before = hash(buf)
            prune!(buf, 1.0f0)
            @test hash(buf) != h_before
        end
    end

    @testset "Property invariants" begin
        rng = MersenneTwister(246)
        N = 100
        atol = 1.0f-5

        @testset "normalize_l1! properties" begin
            for _ in 1:N
                n = rand(rng, 1:16)
                # Mix of positive, negative, and zero entries
                w = Float32.(randn(rng, n) .* 3)
                if rand(rng) < 0.15
                    fill!(w, 0.0f0)
                end
                original = copy(w)
                total = sum(w)
                normalize_l1!(w)
                if total > 0
                    # Near-cancellation of mixed signs can lose Float32 precision;
                    # skip tight sum≈1 when |total| is tiny relative to the vector scale.
                    scale = max(maximum(abs, original), eps(Float32))
                    if abs(total) < 1.0f-3 * scale
                        @test all(isfinite, w)
                    else
                        @test sum(w) ≈ 1.0f0 atol = atol
                        # Proportionality preserved for non-zero total
                        @test all(isapprox.(w .* total, original; atol = atol, rtol = 1.0f-4))
                    end
                else
                    @test w == original
                end
            end
        end

        @testset "normalize_max! properties" begin
            for _ in 1:N
                n = rand(rng, 1:16)
                w = Float32.(randn(rng, n) .* 3)
                # Independent branches: ~15% zeros, ~15% all non-positive (disjoint).
                r = rand(rng)
                if r < 0.15
                    fill!(w, 0.0f0)
                elseif r < 0.30
                    # All non-positive so peak <= 0
                    w .= -abs.(w)
                end
                original = copy(w)
                peak = maximum(w)
                normalize_max!(w)
                if peak > 0
                    @test maximum(w) ≈ 1.0f0 atol = atol
                    @test all(isapprox.(w .* peak, original; atol = atol, rtol = 1.0f-4))
                else
                    @test w == original
                end
            end
        end

        @testset "temporal_weight properties" begin
            for _ in 1:N
                τ = Float32(rand(rng) * 4 + 1.0f-3)  # positive
                dt = Float32(randn(rng) * 5)

                # Symmetry in dt
                @test temporal_weight(dt, τ) ≈ temporal_weight(-dt, τ) atol = atol

                # Closed-form formula
                expected = exp(-abs(dt) / τ)
                @test temporal_weight(dt, τ) ≈ expected atol = atol

                # Zero lag is unity
                @test temporal_weight(0.0f0, τ) ≈ 1.0f0 atol = atol
            end

            # τ <= 0 throws
            for _ in 1:N
                dt = Float32(randn(rng))
                bad_τ = rand(rng) < 0.5 ? 0.0f0 : -Float32(rand(rng) * 5 + eps(Float32))
                @test_throws ArgumentError temporal_weight(dt, bad_τ)
            end
        end

        @testset "prune! properties" begin
            for _ in 1:N
                window = Float32(rand(rng) * 2 + 0.05f0)
                current_time = Float32(rand(rng) * 10)
                n_events = rand(rng, 0:24)
                events = SpikeEvent[
                    SpikeEvent(
                        rand(rng, 1:8),
                        Float32(current_time + (rand(rng) * 4 - 2) * window),
                        Float32(rand(rng) * 2 - 0.5),
                    )
                    for _ in 1:n_events
                ]
                buffer = TemporalBuffer(window, events)
                before_len = length(buffer.events)

                prune!(buffer, current_time)

                # Survivors are within the window
                for event in buffer.events
                    @test (current_time - event.t) <= window + atol
                end

                # Length is nonincreasing
                @test length(buffer.events) <= before_len

                # Every survivor was in the original set (identity by fields)
                survivor_set = Set((e.neuron_id, e.t, e.value) for e in buffer.events)
                original_set = Set((e.neuron_id, e.t, e.value) for e in events)
                @test survivor_set ⊆ original_set

                # Dropped events are outside the window
                for e in events
                    key = (e.neuron_id, e.t, e.value)
                    if key ∉ survivor_set
                        @test (current_time - e.t) > window
                    end
                end

                # Idempotent: second prune leaves events unchanged
                after_first = copy(buffer.events)
                prune!(buffer, current_time)
                @test buffer.events == after_first
            end
        end
        @testset "hash matches ==" begin
            e1 = SpikeEvent(1, 0.1f0, 1.0f0)
            e2 = SpikeEvent(2, 0.2f0, 1.0f0)
            a = SpikeTrain([e1, e2])
            b = SpikeTrain([e1, e2])
            @test hash(a) == hash(b)
            @test length(Set([a, b])) == 1
            @test hash(SpikeEvent(1, 0.5f0, 1.0f0)) == hash(SpikeEvent(1, 0.5f0, 1.0f0))
            @test hash(TemporalBuffer(1.0f0, [e1])) == hash(TemporalBuffer(1.0f0, [e1]))
        end
    end

    # The Experiment Gallery page is generated from experiment artifacts by
    # docs/gallery.jl. It must render correctly from whatever results exist —
    # including none — so the docs build never depends on an experiment having run.
    @testset "Experiment Gallery generator" begin
        include(joinpath(@__DIR__, "..", "docs", "gallery.jl"))
        Gal = Main.Gallery

        function _build(results_dir; repo_root = mktempdir())
            tmp = repo_root
            out = joinpath(tmp, "experiments.md")
            assets = joinpath(tmp, "assets")
            res = Gal.build_gallery(;
                repo_root = tmp,
                results_dir = results_dir,
                out_path = out,
                assets_dir = assets,
            )
            return res, read(out, String), assets
        end

        @testset "no results at all" begin
            res, page, assets = _build(joinpath(mktempdir(), "absent"))

            @test isempty(res.published)
            @test length(res.pending) == length(Gal.ENTRIES)
            @test isempty(res.copied)
            @test !isdir(assets)

            @test startswith(page, "# Experiment Gallery")
            @test occursin("No results published yet", page)
            # Every expected experiment is still introduced, with no results claimed.
            for entry in Gal.ENTRIES
                @test occursin(entry.title, page)
                @test occursin("experiments/$(entry.slug).jl", page)
            end
            # Nothing is illustrated, because nothing has been generated.
            @test !occursin("![", page)
        end

        @testset "published experiment renders generated evidence" begin
            results = joinpath(mktempdir(), "results")
            slug = first(Gal.ENTRIES).slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(dir, "config.toml"),
                "commit = \"abc1234def\"\nseed = 7\n\n[grid]\nn = 5\n")
            write(joinpath(dir, "metrics.csv"),
                "dt,tau,weight\n0.0,1.0,1.0\n1.0,1.0,0.3679\n")
            write(joinpath(dir, "summary.md"),
                "# Result\n\nMax error 1.0e-8.\n\n## Null finding\n\nNo bias observed.\n")
            write(joinpath(dir, "figure.png"), "png-bytes")

            res, page, assets = _build(results)

            @test res.published == [slug]
            @test !(slug in res.pending)
            @test isfile(joinpath(assets, slug, "figure.png"))
            @test occursin("![", page)
            @test occursin("assets/experiments/$(slug)/figure.png", page)
            # Setup comes from config.toml, including nested tables.
            @test occursin("`grid.n`", page)
            @test occursin("`seed`", page)
            # Metrics come from metrics.csv.
            @test occursin("2 recorded rows over 3 columns", page)
            @test occursin("0.3679", page)
            # The generated summary is embedded verbatim, null finding included.
            @test occursin("Max error 1.0e-8.", page)
            @test occursin("No bias observed.", page)
            # Provenance names the code version the run came from.
            @test occursin("abc1234def", page)
            @test !occursin("Provenance incomplete", page)
        end

        @testset "provenance grouped in a nested table" begin
            results = joinpath(mktempdir(), "results")
            slug = first(Gal.ENTRIES).slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(dir, "config.toml"),
                "[params]\ntau = 2.0\n\n[metadata]\ncommit = \"feedface\"\n")

            _, page, _ = _build(results)

            # A commit recorded under a table still counts as provenance, and is
            # not mistaken for a setup parameter.
            @test occursin("`metadata.commit`", page)
            @test !occursin("Provenance incomplete", page)
            @test occursin("`params.tau`", page)
            @test Gal._is_provenance_key("metadata.version")
            @test Gal._is_provenance_key("git.commit")
            @test Gal._is_commit_key("git.commit")
            @test Gal._config_commit(Dict("git" => Dict("commit" => "deadbeef"))) ==
                  "deadbeef"
            @test Gal._is_provenance_key("date")
            @test !Gal._is_provenance_key("kernel.version")
            @test !Gal._is_provenance_key("dataset.date")
            @test !Gal._is_commit_key("kernel.commit")
        end

        @testset "partial artifacts degrade honestly" begin
            results = joinpath(mktempdir(), "results")
            slug = Gal.ENTRIES[2].slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(dir, "metrics.csv"), "kernel,mass\ndiscrete,1.0\n")
            # A created-but-empty directory must not count as a published result.
            mkpath(joinpath(results, Gal.ENTRIES[3].slug))

            res, page, _ = _build(results)

            @test res.published == [slug]
            @test Gal.ENTRIES[3].slug in res.pending
            @test occursin("No `config.toml` was recorded", page)
            @test occursin("No `figure.png` was emitted", page)
            @test occursin("No `summary.md` was emitted", page)
            @test occursin("Provenance incomplete", page)
            @test occursin("| Figure | *not emitted* |", page)
            @test occursin("No checked-in reproduction script is available", page)
            @test !occursin("experiments/$(slug).jl", page)
        end

        @testset "published reproduction commands name existing scripts" begin
            root = mktempdir()
            results = joinpath(root, "experiments", "results")
            slug = first(Gal.ENTRIES).slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(dir, "summary.md"), "Result.\n")
            write(joinpath(root, "experiments", "$(slug).jl"), "# runnable fixture\n")

            _, page, _ = _build(results; repo_root = root)

            @test occursin("julia --project=experiments experiments/$(slug).jl", page)
            @test !occursin("No checked-in reproduction script is available", page)
        end

        @testset "unlisted slugs are still published" begin
            results = joinpath(mktempdir(), "results")
            dir = joinpath(results, "harness_smoke")
            mkpath(dir)
            write(joinpath(dir, "summary.md"), "Smoke artifact.\n")

            res, page, _ = _build(results)

            @test res.published == ["harness_smoke"]
            @test occursin("Additional published results", page)
            @test occursin("Smoke artifact.", page)
        end

        @testset "code provenance and artifact revision are distinct" begin
            root = mktempdir()
            run(`git -C $root init -q`)
            run(`git -C $root config user.email gallery@example.invalid`)
            run(`git -C $root config user.name Gallery-Test`)
            write(joinpath(root, "code.txt"), "experiment code\n")
            run(`git -C $root add code.txt`)
            run(`git -C $root commit -qm code`)
            code_sha = readchomp(`git -C $root rev-parse HEAD`)

            results = joinpath(root, "experiments", "results")
            slug = first(Gal.ENTRIES).slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(dir, "config.toml"), "commit = \"$(code_sha)\"\n")
            write(joinpath(dir, "metrics.csv"), "a\n1\n")
            run(`git -C $root add experiments/results`)
            run(`git -C $root commit -qm artifacts`)
            artifact_sha = readchomp(`git -C $root rev-parse HEAD`)

            _, page, _ = _build(results; repo_root = root)

            # The code commit remains provenance, while links use the later
            # revision that actually contains the generated files.
            @test occursin("/commit/$(code_sha)", page)
            @test occursin("/blob/$(artifact_sha)/", page)
            @test !occursin("/blob/$(code_sha)/", page)
            @test !occursin("/blob/main/", page)
            @test !occursin("Provenance incomplete", page)

            # A dirty regeneration must not be attributed to stale git history.
            write(joinpath(dir, "metrics.csv"), "a\n2\n")
            _, dirty_page, _ = _build(results; repo_root = root)
            @test occursin("Artifact revision incomplete", dirty_page)
            @test occursin("/blob/main/", dirty_page)
            @test !occursin("/blob/$(artifact_sha)/", dirty_page)
        end

        @testset "ignored generated artifacts invalidate git provenance" begin
            root = mktempdir()
            run(`git -C $root init -q`)
            run(`git -C $root config user.email gallery@example.invalid`)
            run(`git -C $root config user.name Gallery-Test`)
            results = joinpath(root, "experiments", "results")
            slug = first(Gal.ENTRIES).slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(root, ".gitignore"), "*.png\n")
            write(joinpath(dir, "config.toml"), "seed = 1\n")
            run(`git -C $root add .gitignore experiments/results`)
            run(`git -C $root commit -qm artifacts`)
            artifact_sha = readchomp(`git -C $root rev-parse HEAD`)
            write(joinpath(dir, "figure.png"), "ignored-regeneration")

            _, page, _ = _build(results; repo_root = root)

            @test occursin("Artifact revision incomplete", page)
            @test occursin("/blob/main/", page)
            @test !occursin("/blob/$(artifact_sha)/", page)
        end

        @testset "unusable commit values are not published as provenance" begin
            results = joinpath(mktempdir(), "results")
            slug = first(Gal.ENTRIES).slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(dir, "config.toml"), "commit = \"unknown\"\n")

            _, page, _ = _build(results)

            @test occursin("Provenance incomplete", page)
            @test occursin("/blob/main/", page)
        end

        @testset "unreadable config is not reported as missing" begin
            results = joinpath(mktempdir(), "results")
            slug = first(Gal.ENTRIES).slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(dir, "config.toml"), "this is = = not toml\n")
            write(joinpath(dir, "metrics.csv"), "a\n1\n")

            _, page, _ = _build(results)

            @test occursin("Unreadable configuration", page)
            @test !occursin("No `config.toml` was recorded", page)
            # The broken file is still linked as evidence.
            @test occursin("config.toml", page)
        end

        @testset "csv records may span lines" begin
            results = joinpath(mktempdir(), "results")
            slug = first(Gal.ENTRIES).slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(dir, "metrics.csv"),
                "kernel,note\ndiscrete,\"first line\nsecond line\"\ntemporal,plain\n")

            _, page, _ = _build(results)

            # A quoted newline is one record, not two.
            @test occursin("2 recorded rows over 2 columns", page)
            @test occursin("first line second line", page)
        end

        @testset "summary embedding is safe" begin
            md = "# Title\n\n```julia\n# not a heading\n```\n\n```@example\n1 + 1\n```\n\n## Sub\n"
            shifted = Gal._shift_headings(md, 4)
            @test occursin("##### Title", shifted)
            @test occursin("###### Sub", shifted)
            # Comments inside fenced code are not treated as headings.
            @test occursin("# not a heading", shifted)
            # Documenter directives in generated summaries must not execute.
            @test !occursin("```@example", shifted)
            @test occursin("```text", shifted)

            # …including directives nested in a block quote or list item, which
            # still open a fenced block in Markdown.
            nested = Gal._shift_headings(
                "> ```@docs\n> x\n> ```\n\n- ```@example\n- x\n- ```\n\n" *
                "- > ```@eval\n- > x\n- > ```\n",
                4,
            )
            @test !occursin("@docs", nested)
            @test !occursin("@example", nested)
            @test !occursin("@eval", nested)

            doctest = Gal._shift_headings(
                "```jldoctest generated\njulia> error(\"must not run\")\n```\n",
                4,
            )
            @test !occursin("```jldoctest", doctest)
            @test occursin("```text", doctest)

            indented_directives = Gal._shift_headings(
                "   ```@eval\n1 + 1\n   ```\n\n10. ```@example\n    2 + 2\n    ```\n",
                4,
            )
            @test !occursin("@eval", indented_directives)
            @test !occursin("@example", indented_directives)
            @test count("```text", indented_directives) == 2

            # A shorter delimiter inside a longer fence is content, not a
            # closer. The later directive must still be neutralized.
            mixed_lengths = Gal._shift_headings(
                "````markdown\n```@example\n````\n\n```@eval\n1 + 1\n```\n",
                4,
            )
            @test occursin("```@example", mixed_lengths)
            @test !occursin("```@eval", mixed_lengths)
            @test occursin("```text", mixed_lengths)

            setext = Gal._shift_headings("Result\n======\n\nDetail\n------\n", 4)
            @test occursin("##### Result", setext)
            @test occursin("###### Detail", setext)
            @test !occursin("======", setext)
            @test !occursin("------", setext)

            indented_atx = Gal._shift_headings("  # Result\n   ## Detail\n", 4)
            @test occursin("  ##### Result", indented_atx)
            @test occursin("   ###### Detail", indented_atx)
        end

        @testset "summary links are rebased to evidence" begin
            root = mktempdir()
            results = joinpath(root, "experiments", "results")
            slug = first(Gal.ENTRIES).slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(dir, "config.toml"), "seed = 1\n")
            write(joinpath(dir, "metrics.csv"), "a\n1\n")
            write(joinpath(dir, "figure.png"), "png")
            write(joinpath(dir, "summary.md"),
                "[details](metrics.csv) ![plot](figure.png) " *
                "[spaced](<figures/result plot.png>) " *
                "[web](https://example.com) [local](#result)\n\n" *
                "[reference][metrics] ![reference plot][figure]\n\n" *
                "[metrics]: metrics.csv?download=1\n" *
                "[figure]: <figure.png> \"generated figure\"\n\n" *
                "![shortcut]\n[shortcut]: figure.png\n\n" *
                "Literal example: `[inline sample](metrics.csv)`.\n\n" *
                "```text\n[code sample](metrics.csv)\n[code-ref]: metrics.csv\n```\n\n" *
                "    [indented sample](metrics.csv)\n")

            _, page, _ = _build(results; repo_root = root)

            @test occursin("[details]($(Gal.REPO_URL)/blob/main/", page)
            @test occursin("![plot](https://raw.githubusercontent.com/rmems/TemporalFocus.jl/main/", page)
            @test occursin("[spaced](<$(Gal.REPO_URL)/blob/main/", page)
            @test occursin("figures/result plot.png>)", page)
            @test occursin("[web](https://example.com)", page)
            @test occursin("[local](#result)", page)
            @test occursin("[metrics]: $(Gal.REPO_URL)/blob/main/", page)
            @test occursin("metrics.csv?download=1", page)
            @test occursin("[figure]: https://raw.githubusercontent.com/rmems/TemporalFocus.jl/main/", page)
            @test occursin("figure.png \"generated figure\"", page)
            @test occursin("[shortcut]: https://raw.githubusercontent.com/rmems/TemporalFocus.jl/main/", page)
            @test occursin("`[inline sample](metrics.csv)`", page)
            @test occursin("[code sample](metrics.csv)", page)
            @test occursin("[code-ref]: metrics.csv", page)
            @test occursin("    [indented sample](metrics.csv)", page)
        end

        @testset "summary leading indentation is preserved" begin
            root = mktempdir()
            results = joinpath(root, "experiments", "results")
            slug = first(Gal.ENTRIES).slug
            dir = joinpath(results, slug)
            mkpath(dir)
            write(joinpath(dir, "summary.md"), "\n    first code line\n    second code line\n")

            _, page, _ = _build(results; repo_root = root)

            @test occursin("    first code line\n    second code line", page)
        end

        @testset "csv parsing" begin
            @test Gal._split_csv_line("a,b,c") == ["a", "b", "c"]
            @test Gal._split_csv_line("a,\"b,c\",d") == ["a", "b,c", "d"]
            @test Gal._split_csv_line("a,\"say \"\"hi\"\"\",c") == ["a", "say \"hi\"", "c"]
            @test Gal._cell("a|b\nc") == "a\\|b c"
        end

        @testset "internal helpers use private names" begin
            for public_style in (:collect_result, :csv_preview, :default_repo_root,
                                 :flatten_config, :render_result, :result_commit,
                                 :shift_headings, :split_csv_line)
                @test !isdefined(Gal, public_style)
            end
            @test isdefined(Gal, :_collect_result)
            @test isdefined(Gal, :_shift_headings)
        end
    end

end
