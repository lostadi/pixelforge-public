(defun coverage-json (logical seen)
  (let ((keys (sort (loop for key being the hash-keys of seen collect key)
                    #'string< :key #'prin1-to-string)))
    (with-output-to-string (out)
      (format out "{\"schema\":\"pixelforge.independent-column-coverage.v1\",\"oracle\":\"Common Lisp from original plan dimensions\",\"logical_tiles\":~D,\"unique_geometries\":~D,\"programs\":[" logical (length keys))
      (loop for key in keys for index from 0 do
        (when (> index 0) (write-char #\, out))
        (format out "{\"kind\":~D,\"rows\":~D,\"physical_columns\":~D,\"logical_uses\":~D}"
                (first key) (second key) (third key) (gethash key seen)))
      (write-string "]}" out))))
(let* ((dense (with-open-file (stream "dense-plan.sexp")
                (let ((*read-eval* nil)) (read stream))))
       (seen (make-hash-table :test #'equal))
       (logical 0))
  (dolist (tensor dense)
    (destructuring-bind (rows columns bank kind) tensor
      (when (member kind '(13 14))
        (unless (and (plusp rows) (plusp columns) (<= 1 bank rows)
                     (zerop (mod columns 256)))
          (error "Invalid pinned original group geometry"))
        (let* ((pieces (ceiling columns 3072))
               (physical (if (> pieces 1) 3072 columns)))
          (loop for start from 0 below rows by bank do
            (let ((height (min bank (- rows start))))
              (dotimes (piece pieces)
                (declare (ignore piece))
                (incf logical)
                (incf (gethash (list kind height physical) seen 0)))))))))
  (let ((result (coverage-json logical seen)))
    (with-open-file (stream "lisp-coverage.json"
                           :direction :output :if-exists :supersede
                           :if-does-not-exist :create)
      (write-line result stream))
    (write-line result)))
