function [y, alpha, obj] = DPMKKM(K, F)
% DPMKKM  Discrete Multiple Kernel k-means.
%   [y, alpha, obj] = DPMKKM(K, F)
%   K: n*n kernel matrix.
%   F: n*c initial label indicator matrix.
%
%   Wang, R., Lu, J., Lu, Y., Nie, F., & Li, X. (2021). Discrete Multiple
%   Kernel k-means. In Proceedings of the Thirtieth International Joint
%   Conference on Artificial Intelligence (Vol. 3, pp. 3111–3117).
%
%   Please contact Rong Wang <wangrong07@tsinghua.org.cn> if you have any
%   questions.
%
%   SPDX-FileCopyrightText: 2020-2021 Jitao Lu <dianlujitao@gmail.com>
%   SPDX-License-Identifier: MIT

numker = size(K, 3);
numclass = size(F, 2);
alpha = zeros(numker, 1) + 1 / numker;
KK = reshape(K, [], numker);
M = KK' * KK;

for iter = 1:10
    % Update F, Fix alpha;
    K_alpha = calculate_kernel_theta(K, alpha);
    F = solve_F(K_alpha, F);
    d = calc_d(K, F);

    obj(iter) = alpha' * M * alpha + alpha' * d + numclass;
    if iter > 2 && (obj(iter - 1) - obj(iter)) / obj(iter - 1) < 1e-9
        break;
    end

    % Update alpha, Fix F;
    A = [];
    b = [];
    Aeq = ones(1, numker);
    beq = 1;
    lb  = zeros(numker,1);
    ub =  ones(numker,1);
    x0 = [];

    alpha = quadprog(M, d, A,b,Aeq,beq,lb,ub, x0, optimset('Display', 'off'));
end

y = vec2ind(F')';

end

function d = calc_d(K, F)
    ff = -2 ./ sum(F);
    KF = shiftdim(sum(F .* pagemtimes(K, F)), 1);
    d = (ff * KF)';
end

function K_alpha = calculate_kernel_theta(K, alpha)
    [num, ~, numker] = size(K);
    KK = reshape(K, [], numker);
    K_alpha = reshape(KK * alpha, num, num);
end

function [F, obj] = solve_F(K, F)
    fKf = sum(F .* (K * F))';
    ff = sum(F)';

    m_all = vec2ind(F')';

    obj(1) = sum(fKf ./ ff);

    for iter = 2:50
        for i = 1:size(F, 1)
            m = m_all(i);
            if ff(m) == 1
                % avoid generating empty cluster
                continue;
            end

            Y_A = F' * K(:, i);

            fKf_s = fKf + 2 * Y_A + K(i, i);
            fKf_s(m) = fKf(m);
            ff_k = ff + 1;
            ff_k(m) = ff(m);

            fKf_0 = fKf;
            fKf_0(m) = fKf(m) - 2 * Y_A(m) + K(i, i);
            ff_0 = ff;
            ff_0(m) = ff(m) - 1;

            delta = fKf_s ./ ff_k - fKf_0 ./ ff_0;

            [~, p] = max(delta);
            if p ~= m
                fKf([m, p]) = [fKf_0(m), fKf_s(p)];
                ff([m, p]) = [ff_0(m), ff_k(p)];

                F(i, [p, m]) = [1, 0];
                m_all(i) = p;
            end
        end
        obj(iter) = sum(fKf ./ ff);

        if iter > 2 && (obj(iter) - obj(iter - 1)) / obj(iter - 1) < 1e-9
            break;
        end
    end
end
